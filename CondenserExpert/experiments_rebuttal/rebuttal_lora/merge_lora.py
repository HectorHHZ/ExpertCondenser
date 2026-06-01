"""Merge a LoRA adapter into its base model and save the full merged model.

The eval script (evaluate_per_layer_r.py) calls AutoModelForCausalLM.from_pretrained,
which does not understand PEFT adapter directories. We therefore merge first and
save the result so eval is a drop-in.

Also copies the moe_bias_states.json / forced_experts_records.json artifacts that
the training script writes alongside the checkpoint.
"""

import argparse
import json
import os
import shutil
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(adapter_dir: str, output_dir: str) -> None:
    adapter_cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(adapter_cfg_path):
        sys.exit(f"ERROR: no adapter_config.json in {adapter_dir}")

    with open(adapter_cfg_path) as f:
        adapter_cfg = json.load(f)
    base_path = adapter_cfg["base_model_name_or_path"]

    print(f"[merge_lora] base = {base_path}", flush=True)
    print(f"[merge_lora] adapter = {adapter_dir}", flush=True)
    print(f"[merge_lora] output = {output_dir}", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    peft = PeftModel.from_pretrained(base, adapter_dir)
    merged = peft.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tok.save_pretrained(output_dir)

    # Carry through aux-free / condenser artifacts if present.
    for fname in ("moe_bias_states.json", "forced_experts_records.json"):
        src = os.path.join(adapter_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, fname))
            print(f"[merge_lora] copied {fname}", flush=True)

    print(f"[merge_lora] done -> {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Directory with adapter_config.json")
    parser.add_argument("--output", required=True, help="Where to save merged full model")
    args = parser.parse_args()
    merge(args.adapter, args.output)
