from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

MODULE_ROOT = Path(__file__).resolve().parent.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

FORCED_EXPERTS_RECORDS = {}

logger = logging.getLogger(__name__)

class AuxFreeOlmoeSparseMoeBlock(OlmoeSparseMoeBlock):
    """
    Auxiliary-free Olmoe MoE implementation using inheritance and in-place modifications.

    This implementation:
    1. Inherits from OlmoeSparseMoeBlock to maintain compatibility
    2. Uses in-place modifications to avoid computational graph issues
    3. Supports forced expert activation
    4. Compatible with DeepSpeed and DDP
    5. Maintains all original Olmoe logic
    """

    def __init__(self, config):
        super().__init__(config)

        self.bias_update_speed = getattr(config, "bias_update_speed", 1e-4)
        self.enable_forced_experts = getattr(config, "enable_forced_experts", False)
        self.num_forced_experts = getattr(config, "num_forced_experts", 2)

        self.register_buffer("bias", torch.zeros(self.num_experts), persistent=True)

        if self.enable_forced_experts:
            self.register_buffer(
                "forced_expert_indices",
                torch.full((self.num_forced_experts,), -1),
                persistent=True,
            )
            self.forced_experts_initialized = False

        logger.info(
            f"✅ Initialized AuxFreeOlmoeSparseMoeBlock: {self.num_experts} experts, "
            f"bias_update_speed={self.bias_update_speed:.6f}, "
            f"forced_experts={self.enable_forced_experts}"
        )

    def select_forced_experts(self, layer_name: Optional[str] = None, highest: bool = False) -> torch.Tensor:
        """
        Select lowest-bias experts as forced activation experts.
        Delayed selection: select during first forward pass to ensure pretrained weights are loaded.

        Args:
            layer_name: Optional layer name for recording
            highest: If True, select highest bias experts instead of lowest

        Returns:
            torch.Tensor: forced expert indices
        """
        if not self.forced_experts_initialized and hasattr(self, "bias"):
            if torch.all(self.bias == 0):
                logger.warning("Bias all zeros, skipping forced expert selection until weights are loaded")
                return self.forced_expert_indices

            if not highest:
                _, lowest_indices = torch.topk(self.bias, self.num_forced_experts, largest=False)
            else:
                _, lowest_indices = torch.topk(self.bias, self.num_forced_experts, largest=True)

            self.forced_expert_indices.copy_(lowest_indices.to(self.forced_expert_indices.device))
            self.forced_experts_initialized = True

            if layer_name:
                FORCED_EXPERTS_RECORDS[layer_name] = {
                    "forced_expert_indices": lowest_indices.cpu().tolist(),
                    "num_forced_experts": self.num_forced_experts,
                    "total_experts": self.num_experts,
                }

            logger.info(f"🎯 Selected forced experts from pretrained bias: {lowest_indices.tolist()}")
            logger.info(f"📊 Selected expert bias values: {self.bias[lowest_indices].tolist()}")
            logger.info(f"📈 Full bias range: [{self.bias.min().item():.6f}, {self.bias.max().item():.6f}]")

        return self.forced_expert_indices

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ✨ DDP-compatible aux-free forward using in-place modification
        Maintains all original Olmoe logic while adding aux-free and forced expert features
        """
        if self.bias.device != hidden_states.device:
            self.bias = self.bias.to(hidden_states.device)

        parent_result, parent_router_logits = super().forward(hidden_states)

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)

        if hasattr(self, "bias"):
            if self.bias.device != router_logits.device:
                self.bias = self.bias.to(router_logits.device)
            router_logits = router_logits + self.bias

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.enable_forced_experts and hasattr(self, "forced_expert_indices"):
            if not self.forced_experts_initialized:
                forced_indices = self.select_forced_experts()
            else:
                forced_indices = self.forced_expert_indices

            if self.forced_experts_initialized and torch.any(forced_indices >= 0):
                batch_size_flat = router_logits.shape[0]

                full_routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                forced_expert_weights = full_routing_weights[:, forced_indices]

                forced_experts_expanded = forced_indices.unsqueeze(0).expand(batch_size_flat, -1)

                final_routing_weights = torch.cat([routing_weights, forced_expert_weights], dim=-1)
                final_selected_experts = torch.cat([selected_experts, forced_experts_expanded], dim=-1)

                routing_weights = final_routing_weights
                selected_experts = final_selected_experts

                if not hasattr(self, "_logged_expert_selection"):
                    logger.info(
                        f"🎯 Enhanced expert selection: original_top_k={self.top_k} + forced_experts={self.num_forced_experts}"
                        f" = total_{self.top_k + self.num_forced_experts}"
                    )
                    self._logged_expert_selection = True

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        parent_result.copy_(final_hidden_states)

        if self.training and hasattr(self, "bias_update_speed") and self.bias_update_speed > 0:
            full_routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            self._expert_usage = full_routing_weights.sum(dim=0).detach()

        return parent_result, parent_router_logits

    def update_bias_after_step(self):
        """Call this after optimizer.step() to update bias"""
        if hasattr(self, "_expert_usage") and self.training:
            with torch.no_grad():
                expert_usage = self._expert_usage

                if dist.is_initialized() and dist.get_world_size() > 1:
                    dist.all_reduce(expert_usage, op=dist.ReduceOp.SUM)
                    expert_usage = expert_usage / dist.get_world_size()

                avg_usage = expert_usage.mean()
                bias_update = torch.zeros_like(self.bias)
                overloaded_mask = expert_usage > avg_usage
                underloaded_mask = expert_usage < avg_usage
                bias_update[overloaded_mask] = +self.bias_update_speed
                bias_update[underloaded_mask] = -self.bias_update_speed
                self.bias.add_(bias_update)

                delattr(self, "_expert_usage")
