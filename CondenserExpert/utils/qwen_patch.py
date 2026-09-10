"""Aux-free routing patch for Qwen MoE (Qwen1.5-MoE / Qwen2-MoE)."""
from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn.functional as F

from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

from .aux_free import AuxFreeMoeMixin

FORCED_EXPERTS_RECORDS: dict = {}

logger = logging.getLogger(__name__)


class AuxFreeQwen2MoeSparseMoeBlock(AuxFreeMoeMixin, Qwen2MoeSparseMoeBlock):
    """
    Auxiliary-free Qwen2 MoE block.

    1. Inherits from Qwen2MoeSparseMoeBlock to maintain compatibility
    2. Uses in-place modifications to avoid computational graph issues
    3. Supports forced expert activation
    4. Compatible with DeepSpeed and DDP
    5. Maintains all original Qwen2 logic

    Unlike the OLMoE block, the bias update is applied immediately at the end
    of every training forward pass (the historical behavior of this patch).
    """

    FORCED_EXPERTS_RECORDS = FORCED_EXPERTS_RECORDS

    def __init__(self, config):
        super().__init__(config)
        self._init_aux_free(config)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """DDP-compatible aux-free forward using in-place modification."""
        self._sync_bias_device(hidden_states)

        parent_result, parent_router_logits = super().forward(hidden_states)

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states_flat)
        router_logits = router_logits + self.bias

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights_topk, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        routing_weights_topk, selected_experts = self._append_forced_experts(
            routing_weights, routing_weights_topk, selected_experts
        )

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

        if self.training and self.bias_update_speed > 0:
            self._bias_update_from_usage(routing_weights.sum(dim=0).detach())

        return parent_result, parent_router_logits
