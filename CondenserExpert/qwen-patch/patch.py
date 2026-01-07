from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

MODULE_ROOT = Path(__file__).resolve().parent.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

FORCED_EXPERTS_RECORDS = {}

logger = logging.getLogger(__name__)

class AuxFreeQwen2MoeSparseMoeBlock(Qwen2MoeSparseMoeBlock):
    """
    Auxiliary-free Qwen2 MoE implementation using inheritance and in-place modifications.

    This implementation:
    1. Inherits from Qwen2MoeSparseMoeBlock to maintain compatibility
    2. Uses in-place modifications to avoid computational graph issues
    3. Supports forced expert activation
    4. Compatible with DeepSpeed and DDP
    5. Maintains all original Qwen2 logic
    """

    def __init__(self, config):
        super().__init__(config)

        # Initialize aux-free parameters
        self.bias_update_speed = getattr(config, "bias_update_speed", 1e-4)
        self.enable_forced_experts = getattr(config, "enable_forced_experts", False)
        self.num_forced_experts = getattr(config, "num_forced_experts", 2)

        # Use register_buffer for DDP compatibility
        self.register_buffer("bias", torch.zeros(self.num_experts), persistent=True)

        # Forced expert tracking
        if self.enable_forced_experts:
            self.register_buffer(
                "forced_expert_indices",
                torch.full((self.num_forced_experts,), -1),
                persistent=True,
            )
            self.forced_experts_initialized = False

        logger.info(
            f"✅ Initialized AuxFreeQwen2MoeSparseMoeBlock: {self.num_experts} experts, "
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
        Maintains all original Qwen2 logic while adding aux-free and forced expert features
        """
        if self.bias.device != hidden_states.device:
            self.bias = self.bias.to(hidden_states.device)

        parent_result, parent_router_logits = super().forward(hidden_states)

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states_flat)

        if hasattr(self, "bias"):
            if self.bias.device != router_logits.device:
                self.bias = self.bias.to(router_logits.device)
            router_logits = router_logits + self.bias

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        routing_weights_topk, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.enable_forced_experts and hasattr(self, "forced_expert_indices"):
            if not self.forced_experts_initialized:
                forced_indices = self.select_forced_experts()
            else:
                forced_indices = self.forced_expert_indices

            if self.forced_experts_initialized and torch.any(forced_indices >= 0):
                batch_size_flat = routing_weights.shape[0]

                forced_expert_weights = routing_weights[:, forced_indices]

                forced_experts_expanded = forced_indices.unsqueeze(0).expand(batch_size_flat, -1)

                final_routing_weights = torch.cat([routing_weights_topk, forced_expert_weights], dim=-1)
                final_selected_experts = torch.cat([selected_experts, forced_experts_expanded], dim=-1)

                routing_weights_topk = final_routing_weights
                selected_experts = final_selected_experts

                if not hasattr(self, "_logged_expert_selection"):
                    logger.info(
                        f"🎯 Enhanced expert selection: original_top_k={self.top_k} + forced_experts={self.num_forced_experts}"
                        f" = total_{self.top_k + self.num_forced_experts}"
                    )
                    self._logged_expert_selection = True

        if self.norm_topk_prob:
            routing_weights_topk /= routing_weights_topk.sum(dim=-1, keepdim=True)
        routing_weights_topk = routing_weights_topk.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            current_state = hidden_states_flat[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights_topk[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states_flat)
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states_flat)) * shared_expert_output
        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        parent_result.copy_(final_hidden_states)

        if self.training and hasattr(self, "bias_update_speed") and self.bias_update_speed > 0:
            with torch.no_grad():
                expert_usage = routing_weights.sum(dim=0)

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

        return parent_result, parent_router_logits


