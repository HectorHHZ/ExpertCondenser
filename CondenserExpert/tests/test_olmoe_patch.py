from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_olmoe_patch_module():
    module_name = "olmoe_patch_module"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = Path(__file__).resolve().parents[1] / "olmoe-patch" / "patch.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_select_forced_experts_tracks_lowest_bias():
    patch = _load_olmoe_patch_module()
    patch.FORCED_EXPERTS_RECORDS.clear()

    block = patch.AuxFreeOlmoeSparseMoeBlock.__new__(patch.AuxFreeOlmoeSparseMoeBlock)
    block.num_forced_experts = 2
    block.num_experts = 4
    block.enable_forced_experts = True
    block.bias = torch.tensor([0.5, -1.5, 0.25, -0.75], dtype=torch.float)
    block.forced_expert_indices = torch.full((block.num_forced_experts,), -1, dtype=torch.long)
    block.forced_experts_initialized = False

    indices = block.select_forced_experts(layer_name="layer0")

    assert block.forced_experts_initialized
    assert set(indices.tolist()) == {1, 3}

    record = patch.FORCED_EXPERTS_RECORDS["layer0"]
    assert set(record["forced_expert_indices"]) == {1, 3}
    assert record["num_forced_experts"] == block.num_forced_experts
    assert record["total_experts"] == block.num_experts
