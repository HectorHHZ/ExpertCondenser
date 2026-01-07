from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_deepseek_patch_module():
    module_name = "deepseek_patch_module"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = Path(__file__).resolve().parents[1] / "deepseek-patch" / "patch.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_select_forced_experts_deepseek_records_lowest_bias():
    patch = _load_deepseek_patch_module()
    patch.FORCED_EXPERTS_RECORDS.clear()

    class DummyGate:
        pass

    gate = DummyGate()
    gate.bias = torch.tensor([-0.2, 1.3, -1.9, 0.4], dtype=torch.float)
    gate.forced_expert_indices = torch.full((2,), -1, dtype=torch.long)
    gate.forced_experts_initialized = False
    gate.n_routed_experts = 4
    gate._layer_name = "gate0"

    indices = patch.select_forced_experts_deepseek(gate, num_forced_experts=2)

    assert gate.forced_experts_initialized
    assert set(indices.tolist()) == {0, 2}

    record = patch.FORCED_EXPERTS_RECORDS["gate0"]
    assert set(record["forced_expert_indices"]) == {0, 2}
    assert record["num_forced_experts"] == 2
    assert record["total_experts"] == gate.n_routed_experts
