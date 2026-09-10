from __future__ import annotations

import torch

from CondenserExpert.utils import deepseek_patch


def test_select_forced_experts_deepseek_records_lowest_bias():
    deepseek_patch.FORCED_EXPERTS_RECORDS.clear()

    class DummyGate:
        pass

    gate = DummyGate()
    gate.bias = torch.tensor([-0.2, 1.3, -1.9, 0.4], dtype=torch.float)
    gate.forced_expert_indices = torch.full((2,), -1, dtype=torch.long)
    gate.forced_experts_initialized = False
    gate.n_routed_experts = 4
    gate._layer_name = "gate0"

    indices = deepseek_patch.select_forced_experts_deepseek(gate, num_forced_experts=2)

    assert gate.forced_experts_initialized
    assert set(indices.tolist()) == {0, 2}

    record = deepseek_patch.FORCED_EXPERTS_RECORDS["gate0"]
    assert set(record["forced_expert_indices"]) == {0, 2}
    assert record["num_forced_experts"] == 2
    assert record["total_experts"] == gate.n_routed_experts
