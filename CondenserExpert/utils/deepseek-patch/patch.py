
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Union

import torch
import torch.distributed as dist
import torch.nn.functional as F

MODULE_ROOT = Path(__file__).resolve().parent.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

FORCED_EXPERTS_RECORDS = {}

logger = logging.getLogger(__name__)


def select_forced_experts_deepseek(
    gate_module,
    num_forced_experts: Union[int, str, None],
    highest: bool = False,
) -> torch.Tensor:
    """Select lowest-bias experts as forced activation experts for DeepSeek V2."""
    if not getattr(gate_module, "forced_experts_initialized", False) and hasattr(gate_module, "bias"):
        if torch.all(gate_module.bias == 0):
            logger.warning("Gate bias all zeros; delay forced expert selection")
            return gate_module.forced_expert_indices

        layer_name = getattr(gate_module, "_layer_name", "unknown_layer")
        if isinstance(num_forced_experts, str):
            layer_name = num_forced_experts
            forced_count = getattr(gate_module, "num_forced_experts", 0)
        elif isinstance(num_forced_experts, int) and num_forced_experts > 0:
            forced_count = num_forced_experts
        else:
            forced_count = getattr(gate_module, "num_forced_experts", 0)

        if forced_count <= 0:
            return gate_module.forced_expert_indices

        _, indices = torch.topk(gate_module.bias, forced_count, largest=highest)
        gate_module.forced_expert_indices.copy_(indices.to(gate_module.forced_expert_indices.device))
        gate_module.forced_experts_initialized = True
        FORCED_EXPERTS_RECORDS[layer_name] = {
            "forced_expert_indices": indices.cpu().tolist(),
            "num_forced_experts": forced_count,
            "total_experts": gate_module.n_routed_experts,
        }
        logger.info("Selected forced experts for %s -> %s", layer_name, indices.tolist())
    return gate_module.forced_expert_indices


def patch_deepseek_model(
    model: torch.nn.Module,
    model_args,
    num_forced_experts: int,
    bias_update_speed: float,
) -> torch.nn.Module:
    """Apply auxiliary-free routing patches to DeepSeek MoE layers."""
    patched = 0

    for name, module in model.named_modules():
        cls_name = module.__class__.__name__

        if cls_name == "DeepseekV2MoE" and getattr(model_args, "remove_aux_loss", False):

            def patched_moe_forward(self, hidden_states):  # type: ignore[override]
                identity = hidden_states
                orig_shape = hidden_states.shape

                topk_idx, topk_weight, _ = self.gate(hidden_states)

                flat = hidden_states.view(-1, hidden_states.shape[-1])
                flat_idx = topk_idx.view(-1)

                if self.training:
                    experts_per_tok = topk_idx.shape[1]
                    expanded = flat.repeat_interleave(experts_per_tok, dim=0)
                    y = torch.empty_like(expanded)
                    for expert_idx, expert in enumerate(self.experts):
                        mask = (flat_idx == expert_idx)
                        if mask.any():
                            y[mask] = expert(expanded[mask])
                    y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
                    y = y.to(flat.dtype).view(*orig_shape)
                else:
                    y = self.moe_infer(flat, topk_idx, topk_weight).view(*orig_shape)

                if getattr(self.config, "n_shared_experts", None) is not None:
                    y = y + self.shared_experts(identity)
                return y

            module.forward = patched_moe_forward.__get__(module, type(module))  # per-instance bind
            patched += 1

        elif cls_name == "MoEGate":
            if getattr(model_args, "add_aux_free_loss", False):
                if not hasattr(module, "bias"):
                    module.bias_update_speed = bias_update_speed
                    module.register_buffer("bias", torch.zeros(module.n_routed_experts), persistent=True)

                if getattr(model_args, "enable_forced_experts", False):
                    module.num_forced_experts = int(num_forced_experts)
                    if not hasattr(module, "forced_expert_indices"):
                        module.register_buffer(
                            "forced_expert_indices",
                            torch.full((num_forced_experts,), -1, dtype=torch.long),  # long!
                            persistent=True,
                        )
                    module.forced_experts_initialized = False
                    module._layer_name = name  # optional for logging

                def patched_gate_forward(self, hidden_states):  # type: ignore[override]
                    bsz, seq_len, hidden = hidden_states.shape
                    flat = hidden_states.view(-1, hidden)

                    logits = F.linear(flat.float(), self.weight.float(), None)
                    if hasattr(self, "bias"):
                        # keep bias on same device
                        logits = logits + self.bias.to(logits.device)

                    if getattr(self, "topk_method", "greedy") == "group_limited_greedy":
                        group_scores = logits.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values
                        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
                        group_mask = torch.zeros_like(group_scores)
                        group_mask.scatter_(1, group_idx, 1)
                        score_mask = (
                            group_mask.unsqueeze(-1)
                            .expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group)
                            .reshape(bsz * seq_len, -1)
                        )
                        masked = logits.masked_fill(~score_mask.bool(), 0.0)
                        topk_weight, topk_idx = torch.topk(
                            masked.softmax(dim=-1, dtype=torch.float32),
                            k=self.top_k,
                            dim=-1,
                            sorted=False,
                        )
                    else:
                        topk_weight, topk_idx = torch.topk(
                            logits.softmax(dim=-1, dtype=torch.float32),
                            k=self.top_k,
                            dim=-1,
                            sorted=False,
                        )

                    if getattr(model_args, "enable_forced_experts", False) and hasattr(self, "forced_expert_indices"):
                        if not getattr(self, "forced_experts_initialized", False):
                            select_forced_experts_deepseek(self, getattr(self, "_layer_name", None))
                        forced = self.forced_expert_indices.to(logits.device)
                        if (forced >= 0).any():
                            batch_n = topk_weight.shape[0]
                            routing = F.softmax(logits, dim=-1, dtype=torch.float32)
                            forced_w = routing[:, forced]
                            forced_i = forced.unsqueeze(0).expand(batch_n, -1)
                            topk_weight = torch.cat([topk_weight, forced_w], dim=-1)
                            topk_idx = torch.cat([topk_idx, forced_i], dim=-1)

                    if self.top_k > 1 and getattr(self, "norm_topk_prob", False):
                        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
                    else:
                        topk_weight = topk_weight * self.routed_scaling_factor

                    if self.training and hasattr(self, "bias") and getattr(self, "bias_update_speed", 0) > 0:
                        with torch.no_grad():
                            rw = F.softmax(logits, dim=-1, dtype=torch.float)
                            usage = rw.sum(dim=0)  # [n_experts]
                            if dist.is_initialized() and dist.get_world_size() > 1:
                                dist.all_reduce(usage, op=dist.ReduceOp.SUM)
                                usage = usage / dist.get_world_size()
                            avg = usage.mean()
                            upd = torch.zeros_like(self.bias)
                            upd[usage > avg] = self.bias_update_speed
                            upd[usage < avg] = -self.bias_update_speed
                            self.bias.add_(upd.to(self.bias.device))

                    return topk_idx, topk_weight, None

                module.forward = patched_gate_forward.__get__(module, type(module))
                patched += 1

    logger.info("Patched %d DeepSeek modules", patched)
    return model
