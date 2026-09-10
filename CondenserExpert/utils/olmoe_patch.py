"""Aux-free routing patch for OLMoE (allenai/OLMoE-*)."""
from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn.functional as F

from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

from .aux_free import AuxFreeMoeMixin

FORCED_EXPERTS_RECORDS: dict = {}

logger = logging.getLogger(__name__)


class AuxFreeOlmoeSparseMoeBlock(AuxFreeMoeMixin, OlmoeSparseMoeBlock):
    """
    Auxiliary-free OLMoE MoE block.

    1. Inherits from OlmoeSparseMoeBlock to maintain compatibility
    2. Uses in-place modifications to avoid computational graph issues
    3. Supports forced expert activation
    4. Compatible with DeepSpeed and DDP
    5. Maintains all original OLMoE logic

    The bias update is deferred: the block records expert usage during the
    forward pass and `update_bias_after_step()` must be invoked once per
    optimizer step (sft_unified wires this up via MoeBiasUpdateCallback).
    """

    FORCED_EXPERTS_RECORDS = FORCED_EXPERTS_RECORDS

    def __init__(self, config):
        super().__init__(config)
        self._init_aux_free(config)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """DDP-compatible aux-free forward using in-place modification."""
        self._sync_bias_device(hidden_states)

        # Run the upstream forward so every expert participates in the autograd
        # graph (keeps DDP/DeepSpeed happy), then overwrite the result in place
        # with the aux-free routing outcome below.
        parent_result, parent_router_logits = super().forward(hidden_states)

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        router_logits = router_logits + self.bias

        full_routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(full_routing_weights, self.top_k, dim=-1)

        routing_weights, selected_experts = self._append_forced_experts(
            full_routing_weights, routing_weights, selected_experts
        )

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

        if self.training and self.bias_update_speed > 0:
            self._record_expert_usage(router_logits)

        return parent_result, parent_router_logits
