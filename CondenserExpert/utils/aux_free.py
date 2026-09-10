"""Shared building blocks for auxiliary-loss-free MoE blocks.

`AuxFreeMoeMixin` factors out the logic that the OLMoE and Qwen2-MoE
patches have in common:

- the routing-bias buffer and its sign-based update rule,
- delayed selection of forced ("condenser") experts from the loaded bias,
- appending forced experts to the per-token routing decision.

Model-family patches inherit from the mixin plus the upstream block and
keep only the family-specific forward pass.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AuxFreeMoeMixin:
    """Aux-free routing state and helpers shared across model families.

    Expects the concrete class to define ``self.num_experts`` before
    ``_init_aux_free`` is called (upstream MoE blocks all do).
    """

    FORCED_EXPERTS_RECORDS: Dict[str, dict] = {}

    def _init_aux_free(self, config) -> None:
        self.bias_update_speed = getattr(config, "bias_update_speed", 1e-4)
        self.enable_forced_experts = getattr(config, "enable_forced_experts", False)
        self.num_forced_experts = getattr(config, "num_forced_experts", 2)

        self.register_buffer("bias", torch.zeros(self.num_experts), persistent=True)

        if self.enable_forced_experts:
            self.register_buffer(
                "forced_expert_indices",
                torch.full((self.num_forced_experts,), -1, dtype=torch.long),
                persistent=True,
            )
            self.forced_experts_initialized = False

        logger.info(
            "Initialized %s: %d experts, bias_update_speed=%.6f, forced_experts=%s",
            type(self).__name__, self.num_experts, self.bias_update_speed, self.enable_forced_experts,
        )

    # ------------------------------------------------------------------ forced experts
    def select_forced_experts(self, layer_name: Optional[str] = None, highest: bool = False) -> torch.Tensor:
        """Select lowest-bias experts as forced activation experts.

        Delayed selection: runs during the first forward pass so that
        pretrained bias values are already loaded.
        """
        if not self.forced_experts_initialized and hasattr(self, "bias"):
            if torch.all(self.bias == 0):
                logger.warning("Bias all zeros, skipping forced expert selection until weights are loaded")
                return self.forced_expert_indices

            _, indices = torch.topk(self.bias, self.num_forced_experts, largest=highest)

            self.forced_expert_indices.copy_(indices.to(self.forced_expert_indices.device))
            self.forced_experts_initialized = True

            if layer_name:
                self.FORCED_EXPERTS_RECORDS[layer_name] = {
                    "forced_expert_indices": indices.cpu().tolist(),
                    "num_forced_experts": self.num_forced_experts,
                    "total_experts": self.num_experts,
                }

            logger.info("Selected forced experts from pretrained bias: %s", indices.tolist())
            logger.info("Selected expert bias values: %s", self.bias[indices].tolist())
            logger.info(
                "Full bias range: [%.6f, %.6f]", self.bias.min().item(), self.bias.max().item()
            )

        return self.forced_expert_indices

    def _append_forced_experts(
        self,
        full_routing_weights: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate forced experts (and their softmax weights) to the top-k choice."""
        if not (self.enable_forced_experts and hasattr(self, "forced_expert_indices")):
            return routing_weights, selected_experts

        if not self.forced_experts_initialized:
            forced_indices = self.select_forced_experts()
        else:
            forced_indices = self.forced_expert_indices

        if self.forced_experts_initialized and torch.any(forced_indices >= 0):
            n_tokens = full_routing_weights.shape[0]
            forced_weights = full_routing_weights[:, forced_indices]
            forced_expanded = forced_indices.unsqueeze(0).expand(n_tokens, -1)

            routing_weights = torch.cat([routing_weights, forced_weights], dim=-1)
            selected_experts = torch.cat([selected_experts, forced_expanded], dim=-1)

            if not hasattr(self, "_logged_expert_selection"):
                logger.info(
                    "Enhanced expert selection: original_top_k=%d + forced_experts=%d = total_%d",
                    self.top_k, self.num_forced_experts, self.top_k + self.num_forced_experts,
                )
                self._logged_expert_selection = True

        return routing_weights, selected_experts

    # ------------------------------------------------------------------ bias updates
    def _sync_bias_device(self, reference: torch.Tensor) -> None:
        if self.bias.device != reference.device:
            self.bias = self.bias.to(reference.device)

    def _bias_update_from_usage(self, expert_usage: torch.Tensor) -> None:
        """Sign-based aux-free update: raise bias of overloaded experts,
        lower bias of underloaded ones (pushes long-tailed experts toward
        inactivity, per the ExpertCondenser method)."""
        with torch.no_grad():
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(expert_usage, op=dist.ReduceOp.SUM)
                expert_usage = expert_usage / dist.get_world_size()

            avg_usage = expert_usage.mean()
            bias_update = torch.zeros_like(self.bias)
            bias_update[expert_usage > avg_usage] = +self.bias_update_speed
            bias_update[expert_usage < avg_usage] = -self.bias_update_speed
            self.bias.add_(bias_update)

    def _record_expert_usage(self, router_logits: torch.Tensor) -> None:
        """Stash soft expert usage for a deferred update (see update_bias_after_step)."""
        full = F.softmax(router_logits, dim=1, dtype=torch.float)
        self._expert_usage = full.sum(dim=0).detach()

    def update_bias_after_step(self) -> None:
        """Apply the deferred bias update; call once per optimizer step."""
        if hasattr(self, "_expert_usage") and self.training:
            self._bias_update_from_usage(self._expert_usage)
            delattr(self, "_expert_usage")
