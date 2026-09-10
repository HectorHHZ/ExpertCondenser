from __future__ import annotations

import torch

from CondenserExpert.utils import qwen_patch


def test_select_forced_experts_tracks_lowest_bias():
    qwen_patch.FORCED_EXPERTS_RECORDS.clear()

    block = qwen_patch.AuxFreeQwen2MoeSparseMoeBlock.__new__(qwen_patch.AuxFreeQwen2MoeSparseMoeBlock)
    block.num_forced_experts = 2
    block.num_experts = 4
    block.enable_forced_experts = True
    block.bias = torch.tensor([0.5, -1.5, 0.25, -0.75], dtype=torch.float)
    block.forced_expert_indices = torch.full((block.num_forced_experts,), -1, dtype=torch.long)
    block.forced_experts_initialized = False

    indices = block.select_forced_experts(layer_name="layer0")

    assert block.forced_experts_initialized
    assert set(indices.tolist()) == {1, 3}

    record = qwen_patch.FORCED_EXPERTS_RECORDS["layer0"]
    assert set(record["forced_expert_indices"]) == {1, 3}
    assert record["num_forced_experts"] == block.num_forced_experts
    assert record["total_experts"] == block.num_experts
