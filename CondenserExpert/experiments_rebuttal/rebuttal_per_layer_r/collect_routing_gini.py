"""
Path 2: collect per-layer routing-Gini from the Phase-1 model on math7k.

For each MoE layer:
  1. Hook the gate's forward to capture router_logits.
  2. Add the trained bias buffer (same as Phase-2 routing).
  3. Softmax + top-k --> selected expert indices.
  4. Accumulate per-expert counts.
Then compute Gini per layer and write routing_stats.json.

Run on 1 GPU; ~5 min wall-clock.
"""

import os
import re
import sys
import json
import torch
import numpy as np
from collections import defaultdict

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "src", "open_r1"))

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen2_moe.modeling_qwen2_moe as qwen2_moe_module
from distill_models.qwen2_moe.modeling_qwen2_moe_aux_free_inheritance import (
    AuxFreeQwen2MoeSparseMoeBlock,
)

qwen2_moe_module.Qwen2MoeSparseMoeBlock = AuxFreeQwen2MoeSparseMoeBlock

MODEL_ID = "Anonymous/Qwen1.5-MOE-aux-free-sft-math7k-1e-3-gamma-1epoch"
N_SAMPLES = 500   # math7k inference samples
MAX_LEN = 512
BATCH_SIZE = 8


def gini(counts):
    """Classical Gini on non-negative counts."""
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    cumx = np.cumsum(x)
    return (n + 1 - 2 * cumx.sum() / cumx[-1]) / n


def main():
    print(f"[gini] Loading {MODEL_ID} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Load Phase-1 bias into the model's bias buffers (so routing matches Phase-2).
    from open_r1.sft_aux_new_bias_tracker import load_moe_bias_states
    load_moe_bias_states(model, MODEL_ID)
    print("[gini] Bias loaded.", flush=True)

    # Register forward hooks on each gate.
    num_experts = model.config.num_experts
    top_k = model.config.num_experts_per_tok
    counts = defaultdict(lambda: torch.zeros(num_experts, dtype=torch.long))
    moe_modules = []
    for name, module in model.named_modules():
        if isinstance(module, AuxFreeQwen2MoeSparseMoeBlock):
            m = re.search(r"layers\.(\d+)\.", name)
            if not m:
                continue
            lidx = int(m.group(1))
            moe_modules.append((lidx, name, module))
    moe_modules.sort(key=lambda t: t[0])
    print(f"[gini] Hooking {len(moe_modules)} MoE blocks.", flush=True)

    def make_gate_hook(lidx, bias_buf):
        def hook(mod, inp, out):
            # out is router_logits  [tokens, num_experts]
            biased = out + bias_buf.to(out.dtype).to(out.device)
            w = torch.softmax(biased, dim=1, dtype=torch.float32)
            _, selected = torch.topk(w, top_k, dim=-1)
            cnt = torch.bincount(
                selected.flatten().to(torch.long),
                minlength=num_experts,
            ).cpu()
            counts[lidx] += cnt
        return hook

    for lidx, _name, module in moe_modules:
        module.gate.register_forward_hook(make_gate_hook(lidx, module.bias))

    # Load math7k and run a sample
    from datasets import load_dataset
    print("[gini] Loading math7k ...", flush=True)
    ds = load_dataset("Anonymous/math7k")["train"].select(range(N_SAMPLES))
    prompts = [ex["instruction"] for ex in ds]

    print(f"[gini] Running inference on {N_SAMPLES} samples (bs={BATCH_SIZE}, max_len={MAX_LEN})...", flush=True)
    with torch.no_grad():
        for i in range(0, N_SAMPLES, BATCH_SIZE):
            batch = prompts[i : i + BATCH_SIZE]
            toks = tokenizer(
                batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"
            ).to(model.device)
            _ = model(**toks)
            if (i // BATCH_SIZE) % 10 == 0:
                print(f"  batch {i // BATCH_SIZE + 1} / {(N_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE}", flush=True)

    # Compute Gini per layer.
    per_layer = {}
    for lidx in sorted(counts):
        c = counts[lidx].numpy()
        per_layer[lidx] = {
            "counts": c.tolist(),
            "gini": float(gini(c)),
            "total_selections": int(c.sum()),
            "max_count": int(c.max()),
            "min_count": int(c.min()),
            "n_zero": int((c == 0).sum()),
        }

    out_path = os.path.join(os.path.dirname(__file__), "routing_stats.json")
    with open(out_path, "w") as f:
        json.dump(per_layer, f, indent=2)
    print(f"[gini] Wrote {out_path}", flush=True)

    # Pretty print
    print()
    print(f"{'L':>3} {'gini':>8} {'max':>6} {'zeros':>6}")
    for lidx in sorted(per_layer):
        d = per_layer[lidx]
        print(f"{lidx:>3} {d['gini']:>8.4f} {d['max_count']:>6d} {d['n_zero']:>6d}")


if __name__ == "__main__":
    main()
