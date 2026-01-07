"""Utilities for managing aux-free MoE bias state checkpoints."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _resolve_bias_file_path(model_path_or_repo: str) -> Optional[str]:
    """Return the resolved path to `moe_bias_states.json`, downloading if required."""
    # Import on demand so environments without HF Hub remain valid when loading from disk.
    try:
        if "/" in model_path_or_repo and not os.path.exists(model_path_or_repo):
            from huggingface_hub import hf_hub_download  # type: ignore

            logger.info(
                "Downloading moe_bias_states.json from HuggingFace Hub: %s", model_path_or_repo
            )
            return hf_hub_download(
                repo_id=model_path_or_repo,
                filename="moe_bias_states.json",
                cache_dir=None,
            )
    except Exception:  # pragma: no cover - logging handled by caller
        logger.exception("Failed to download MoE bias states from Hub")
        return None

    path = os.path.join(model_path_or_repo, "moe_bias_states.json")
    if os.path.exists(path):
        return path

    logger.warning("moe_bias_states.json not found at: %s", path)
    return None


def load_moe_bias_states(model: torch.nn.Module, model_path_or_repo: str) -> bool:
    """Load MoE bias tensors from a JSON payload and copy them into the model."""
    bias_file_path = _resolve_bias_file_path(model_path_or_repo)
    if bias_file_path is None:
        return False

    try:
        with open(bias_file_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        logger.exception("Error reading MoE bias state file: %s", bias_file_path)
        return False

    bias_states = payload.get("moe_bias_states")
    if not isinstance(bias_states, dict):
        logger.error("Invalid bias file format: `moe_bias_states` missing or incorrect type")
        return False

    loaded_count = 0
    for module_name, module in model.named_modules():
        if not hasattr(module, "bias"):
            continue
        bias_info = bias_states.get(module_name)
        if bias_info is None:
            continue

        try:
            bias_values = torch.tensor(bias_info["bias_values"], dtype=module.bias.dtype)
            module.bias.data.copy_(bias_values.to(module.bias.device))
            loaded_count += 1
            logger.info("Loaded bias for %s: %d experts", module_name, len(bias_values))
        except Exception:
            logger.exception("Failed to load bias for module %s", module_name)

    if loaded_count:
        logger.info("Successfully loaded bias for %d MoE layers", loaded_count)
        return True

    logger.warning("No MoE bias states were loaded from %s", bias_file_path)
    return False


def save_moe_bias_states(
    model: torch.nn.Module,
    output_dir: str,
    filename: str = "moe_bias_states.json",
) -> None:
    """Persist the bias tensors of aux-free MoE layers to a JSON file."""
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    bias_states = {}
    for module_name, module in model.named_modules():
        if hasattr(module, "bias") and hasattr(module, "bias_update_speed"):
            bias_states[module_name] = {
                "bias_values": module.bias.detach().cpu().tolist(),
                "bias_update_speed": float(module.bias_update_speed),
                "num_experts": len(module.bias),
                "module_type": type(module).__name__,
                "device": str(module.bias.device),
                "dtype": str(module.bias.dtype),
            }

    if not bias_states:
        logger.warning("No MoE bias states discovered; skip saving")
        return

    payload = {
        "metadata": {
            "total_moe_layers": len(bias_states),
            "save_timestamp": datetime.now().isoformat(),
            "model_type": type(model).__name__,
        },
        "moe_bias_states": bias_states,
    }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    except Exception:
        logger.exception("Failed to save MoE bias states to %s", path)
        return

    logger.info("Saved MoE bias states to %s", path)
    for layer_name, info in bias_states.items():
        bias_values = info["bias_values"]
        bias_range = f"[{min(bias_values):.4f}, {max(bias_values):.4f}]"
        logger.info(
            "  %s: %d experts, range=%s", layer_name, info["num_experts"], bias_range
        )
