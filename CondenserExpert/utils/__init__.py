from __future__ import annotations

import types
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from .import_utils import is_e2b_available
from .model_utils import get_tokenizer, memory_stats
from .moe_utils import load_moe_bias_states, save_moe_bias_states


def _load_patch_module(name: str, relative_path: str) -> types.ModuleType:
    """Load a patch module living in a hyphenated directory."""
    module_path = Path(__file__).resolve().parent / relative_path
    spec = spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load patch module {name} from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_deepseek_patch = _load_patch_module("open_r1.utils.deepseek_patch", "deepseek-patch/patch.py")
_olmoe_patch = _load_patch_module("open_r1.utils.olmoe_patch", "olmoe-patch/patch.py")
_qwen_patch = _load_patch_module("open_r1.utils.qwen_patch", "qwen-patch/patch.py")

patch_deepseek_model = _deepseek_patch.patch_deepseek_model
DEEPSEEK_FORCED_EXPERTS_RECORDS = _deepseek_patch.FORCED_EXPERTS_RECORDS
select_forced_experts_deepseek = _deepseek_patch.select_forced_experts_deepseek

AuxFreeOlmoeSparseMoeBlock = _olmoe_patch.AuxFreeOlmoeSparseMoeBlock
OLMOE_FORCED_EXPERTS_RECORDS = _olmoe_patch.FORCED_EXPERTS_RECORDS

AuxFreeQwen2MoeSparseMoeBlock = _qwen_patch.AuxFreeQwen2MoeSparseMoeBlock
QWEN_FORCED_EXPERTS_RECORDS = _qwen_patch.FORCED_EXPERTS_RECORDS


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
    "AuxFreeOlmoeSparseMoeBlockSinkhorn",
    "OLMOE_FORCED_EXPERTS_RECORDS",
    "AuxFreeQwen2MoeSparseMoeBlock",
    "AuxFreeQwen2MoeSparseMoeBlockSinkhorn",
    "QWEN_FORCED_EXPERTS_RECORDS",
]
