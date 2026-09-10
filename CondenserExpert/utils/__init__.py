from __future__ import annotations

from .deepseek_patch import (
    FORCED_EXPERTS_RECORDS as DEEPSEEK_FORCED_EXPERTS_RECORDS,
    patch_deepseek_model,
    select_forced_experts_deepseek,
)
from .import_utils import is_e2b_available
from .model_utils import get_tokenizer, memory_stats
from .moe_utils import load_moe_bias_states, save_moe_bias_states
from .olmoe_patch import (
    FORCED_EXPERTS_RECORDS as OLMOE_FORCED_EXPERTS_RECORDS,
    AuxFreeOlmoeSparseMoeBlock,
)
from .qwen_patch import (
    FORCED_EXPERTS_RECORDS as QWEN_FORCED_EXPERTS_RECORDS,
    AuxFreeQwen2MoeSparseMoeBlock,
)

__all__ = [
    "get_tokenizer",
    "is_e2b_available",
    "memory_stats",
    "load_moe_bias_states",
    "save_moe_bias_states",
    "patch_deepseek_model",
    "DEEPSEEK_FORCED_EXPERTS_RECORDS",
    "select_forced_experts_deepseek",
    "AuxFreeOlmoeSparseMoeBlock",
    "OLMOE_FORCED_EXPERTS_RECORDS",
    "AuxFreeQwen2MoeSparseMoeBlock",
    "QWEN_FORCED_EXPERTS_RECORDS",
]
