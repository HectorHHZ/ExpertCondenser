# Rebuttal Experiment A: Per-layer-r ablation for ExpertCondenser.
#
# Wraps src/open_r1/sft_aux_new_bias_tracker.py without modifying it.
# Reads env var PER_LAYER_R_GROUPS (e.g. "4,2,0") = (r_early, r_mid, r_late)
# split across thirds of the MoE layer stack of Qwen1.5-MoE.
#
# Launch (8x H100, ZeRO-2):
#   PER_LAYER_R_GROUPS="4,2,0" \
#   accelerate launch --main_process_port $PORT \
#       --config_file=recipes/accelerate_configs/zero2.yaml \
#       experiments/rebuttal_per_layer_r/sft_aux_layer_ablation.py \
#       --config experiments/rebuttal_per_layer_r/configs/front_4_2_0.yaml

import os
import re
import sys
import logging
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
# Match the original script layout: when src/open_r1/X.py is launched directly,
# sys.path[0] is src/open_r1 — so `from distill_models import ...` resolves.
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "open_r1"))

# IMPORTANT: import the bias_tracker variant (this is what the user runs).
import open_r1.sft_aux_new_bias_tracker as sft_module
from open_r1.sft_aux_new_bias_tracker import main, AuxFreeModelConfig
from open_r1.configs import SFTConfig
from trl import ScriptArguments, TrlParser

logger = logging.getLogger(__name__)

PER_LAYER_R_GROUPS = os.environ.get("PER_LAYER_R_GROUPS", "").strip()


def _compute_layer_r(num_layers: int, parts):
    """Split [0, num_layers) into 3 contiguous groups, assign r per group."""
    g = num_layers // 3
    sizes = [g, g, num_layers - 2 * g]
    layer_r = []
    for r, size in zip(parts, sizes):
        layer_r.extend([r] * size)
    assert len(layer_r) == num_layers
    return layer_r


def apply_per_layer_forced_experts(model, groups_str: str):
    """Locate aux-free MoE blocks, (re-)register `forced_expert_indices`
    buffer per-layer with size r_layer, and set `num_forced_experts` /
    `enable_forced_experts` accordingly.

    Handles two situations:
      (a) Qwen1.5 path: training script does NOT register the buffer (it's
          only registered conditionally in __init__ based on the saved
          config, which usually has enable_forced_experts=False).
      (b) DeepSeek path: training script DOES register a uniform buffer;
          we resize it per layer.
    """
    parts = [int(x) for x in groups_str.split(",")]

    # Locate candidates: any module that already exposes the aux-free hooks.
    candidates = []
    for name, module in model.named_modules():
        is_qwen_block = hasattr(module, "enable_forced_experts") and hasattr(module, "num_experts")
        is_dsv2_gate = hasattr(module, "enable_forced_experts") and hasattr(module, "n_routed_experts")
        if not (is_qwen_block or is_dsv2_gate):
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        if m is None:
            continue
        candidates.append((int(m.group(1)), name, module))

    if not candidates:
        raise RuntimeError(
            "apply_per_layer_forced_experts: no aux-free MoE modules found. "
            "Expected AuxFreeQwen2MoeSparseMoeBlock (with enable_forced_experts attr) "
            "or DeepSeek MoEGate (with n_routed_experts attr)."
        )

    candidates.sort(key=lambda t: t[0])
    num_layers = candidates[-1][0] + 1

    # Two accepted formats:
    #   - 3 ints -> early/mid/late group split
    #   - L ints -> direct per-layer specification
    if len(parts) == 3:
        layer_r = _compute_layer_r(num_layers, parts)
    elif len(parts) == num_layers:
        layer_r = list(parts)
    else:
        raise ValueError(
            f"PER_LAYER_R_GROUPS must have 3 ints (group split) or {num_layers} "
            f"ints (per-layer); got {len(parts)} ints: {groups_str!r}"
        )

    print(
        f"[per-layer-r] groups={parts}  num_layers={num_layers}  layer_r={layer_r}",
        flush=True,
    )

    # Reference device/dtype from any parameter in the model.
    ref_param = next(iter(model.parameters()))
    ref_device = ref_param.device

    for layer_idx, name, module in candidates:
        r = layer_r[layer_idx]
        if r == 0:
            module.enable_forced_experts = False
            # Drop any existing buffer to free a few bytes.
            if hasattr(module, "forced_expert_indices"):
                try:
                    del module.forced_expert_indices
                except AttributeError:
                    pass
            print(f"[per-layer-r] {name} (L{layer_idx}): DISABLED (r=0)")
            continue

        # Drop old buffer if present (may exist from training-script init).
        if hasattr(module, "forced_expert_indices"):
            try:
                old_device = module.forced_expert_indices.device
            except (AttributeError, RuntimeError):
                old_device = ref_device
            try:
                del module.forced_expert_indices
            except AttributeError:
                pass
        else:
            old_device = ref_device

        module.register_buffer(
            "forced_expert_indices",
            torch.full((r,), -1, device=old_device, dtype=torch.long),
            persistent=True,
        )
        module.num_forced_experts = r
        module.enable_forced_experts = True
        module.forced_experts_initialized = False
        print(f"[per-layer-r] {name} (L{layer_idx}): r={r}")


_orig_load = sft_module.load_moe_bias_states


def _patched_load(model, *args, **kwargs):
    print(f"[per-layer-r WRAPPER] _patched_load called  PER_LAYER_R_GROUPS={PER_LAYER_R_GROUPS!r}", flush=True)
    result = _orig_load(model, *args, **kwargs)
    print(f"[per-layer-r WRAPPER] orig load_moe_bias_states returned: {result}", flush=True)
    if PER_LAYER_R_GROUPS:
        apply_per_layer_forced_experts(model, PER_LAYER_R_GROUPS)
        print(f"[per-layer-r WRAPPER] per-layer override applied", flush=True)
    else:
        print(f"[per-layer-r WRAPPER] PER_LAYER_R_GROUPS not set; using uniform num_forced_experts from YAML.", flush=True)
    return result


sft_module.load_moe_bias_states = _patched_load
print(f"[per-layer-r WRAPPER] monkey-patched sft_module.load_moe_bias_states (id={id(sft_module.load_moe_bias_states)})  PER_LAYER_R_GROUPS={PER_LAYER_R_GROUPS!r}", flush=True)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, AuxFreeModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
