"""Real expert-parallelism microbenchmark for a single MoE layer.

Layout (mirrors Qwen1.5-MoE-A2.7B MoE layer dimensions, with n rounded to 64 so the
expert pool partitions evenly across world_size = 8):

  HIDDEN = 2048
  INTERMEDIATE = 1408
  TOTAL_ROUTED_EXPERTS = 64
  TOP_K = 4
  NUM_CONDENSER = 2     (only for the two ExpertCondenser modes)

EP layout:
  Each rank holds (TOTAL_ROUTED_EXPERTS // world_size) = 8 routed experts.
  For ExpertCondenser modes, each rank ALSO instantiates a fresh copy of the
  NUM_CONDENSER condenser experts (true replication — no cross-device traffic).

Per-step workload:
  Each rank generates a fresh random hidden tensor [T, HIDDEN], runs one forward
  through the EP MoE layer, sums the output as a synthetic loss, and runs
  backward. Per-component timings:
    - t_route       : router + select_experts
    - t_dispatch    : all_to_all_single hidden states + payloads to expert hosts
    - t_compute     : local routed-expert FFNs
    - t_combine     : all_to_all_single back to the original ranks
    - t_condenser   : NUM_CONDENSER replicated FFNs (local, no comm) — EC only
    - t_total       : full forward + backward end-to-end

Per-step communication accounting:
    - dispatch_bytes : sum over ranks of hidden_bytes sent OUT to remote ranks
                       (i.e. excluding rank-to-self transfers, which never
                       traverse the NIC/NVLink in practice)
    - combine_bytes  : same, for the return path

Modes (---mode):
  sft               : top-k=4 routed                     (4 routed, 0 condenser)
  densemixer        : all 64 routed active per token     (64 routed, 0 condenser)
  ec_paper          : Eq. 2: (k-r)=2 routed + r=2 cond.  (2 routed, 2 condenser)
  ec_code           : codebase additive: k=4 + r=2 cond. (4 routed, 2 condenser)

Launch (8 GPUs):
  torchrun --nproc_per_node=8 ep_microbenchmark.py --mode sft   --output_dir results_ep
  torchrun --nproc_per_node=8 ep_microbenchmark.py --mode densemixer   --output_dir results_ep
  torchrun --nproc_per_node=8 ep_microbenchmark.py --mode ec_paper     --output_dir results_ep
  torchrun --nproc_per_node=8 ep_microbenchmark.py --mode ec_code      --output_dir results_ep
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# --- Qwen1.5-MoE-A2.7B-shaped MoE layer dimensions ----------------------------
HIDDEN = 2048
INTERMEDIATE = 1408
TOTAL_ROUTED_EXPERTS = 64  # rounded up from 60 to partition evenly across 8 ranks
TOP_K = 4
NUM_CONDENSER = 2

MODES = ("sft", "densemixer", "ec_paper", "ec_code")


# ---------------------------------------------------------------------------- #
# Autograd-aware all-to-all (variable-size)                                    #
# ---------------------------------------------------------------------------- #
class _AllToAllSingleAutograd(torch.autograd.Function):
    """Variable-size all_to_all_single with backward = transposed all_to_all_single.

    Saved between fwd/bwd: input_split_sizes, output_split_sizes, group.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, input_split: List[int],
                output_split: List[int], group=None):
        ctx.input_split = input_split
        ctx.output_split = output_split
        ctx.group = group
        out = torch.empty(sum(output_split), *x.shape[1:], dtype=x.dtype, device=x.device)
        dist.all_to_all_single(
            out, x.contiguous(),
            output_split_sizes=output_split,
            input_split_sizes=input_split,
            group=group,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # transpose splits in backward
        grad_in = torch.empty(sum(ctx.input_split), *grad_out.shape[1:],
                              dtype=grad_out.dtype, device=grad_out.device)
        dist.all_to_all_single(
            grad_in, grad_out.contiguous(),
            output_split_sizes=ctx.input_split,
            input_split_sizes=ctx.output_split,
            group=ctx.group,
        )
        return grad_in, None, None, None


def all_to_all_single_autograd(x, input_split, output_split, group=None):
    return _AllToAllSingleAutograd.apply(x, input_split, output_split, group)


# ---------------------------------------------------------------------------- #
# Expert FFN (Qwen2-style SwiGLU)                                              #
# ---------------------------------------------------------------------------- #
class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj   = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------- #
# Single-layer EP MoE                                                           #
# ---------------------------------------------------------------------------- #
class EPMoELayer(nn.Module):
    def __init__(self, mode: str, dtype=torch.bfloat16):
        super().__init__()
        assert mode in MODES, f"unknown mode {mode}"
        self.mode = mode
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        assert TOTAL_ROUTED_EXPERTS % self.world_size == 0
        self.local_n = TOTAL_ROUTED_EXPERTS // self.world_size

        self.local_experts = nn.ModuleList([Expert() for _ in range(self.local_n)])
        self.gate = nn.Linear(HIDDEN, TOTAL_ROUTED_EXPERTS, bias=False)

        # Replicated condenser experts (the canonical EP deployment)
        if mode in ("ec_paper", "ec_code"):
            self.condenser = nn.ModuleList([Expert() for _ in range(NUM_CONDENSER)])
            # Separate gate for condenser weights (also replicated)
            self.cond_gate = nn.Linear(HIDDEN, NUM_CONDENSER, bias=False)
        else:
            self.condenser = None
            self.cond_gate = None

        self.to(dtype)

    # ---------------- routing ----------------
    def select_routed(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        soft = F.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        if self.mode == "sft":
            w, idx = torch.topk(soft, TOP_K, dim=-1)
        elif self.mode == "densemixer":
            T = logits.shape[0]
            w = soft
            idx = torch.arange(TOTAL_ROUTED_EXPERTS, device=logits.device).unsqueeze(0).expand(T, -1)
        elif self.mode == "ec_paper":
            # (k-r) routed experts. Condenser experts are separate FFNs.
            w, idx = torch.topk(soft, TOP_K - NUM_CONDENSER, dim=-1)
        elif self.mode == "ec_code":
            # k routed + r condenser (additive)
            w, idx = torch.topk(soft, TOP_K, dim=-1)
        else:
            raise ValueError(self.mode)
        return w, idx

    # ---------------- dispatch + expert compute + combine ----------------
    def dispatch_compute_combine(self, x: torch.Tensor, weights: torch.Tensor,
                                 indices: torch.Tensor,
                                 timings: dict, comm_bytes: dict):
        """Returns combined routed output [T, HIDDEN]. Mutates timings/comm_bytes."""
        T = x.shape[0]
        num_active = indices.shape[1]
        N = T * num_active
        device = x.device

        flat_idx = indices.reshape(-1)
        flat_w = weights.reshape(-1)
        tok_id = torch.arange(T, device=device).unsqueeze(1).expand(-1, num_active).reshape(-1)

        target_rank = flat_idx // self.local_n
        local_exp = flat_idx % self.local_n

        # Sort pairs by target rank so each rank's outgoing block is contiguous
        order = torch.argsort(target_rank, stable=True)
        sorted_target = target_rank[order]
        sorted_tok = tok_id[order]
        sorted_local_exp = local_exp[order]
        sorted_w = flat_w[order]

        send_counts = torch.bincount(sorted_target, minlength=self.world_size).to(torch.int64)
        recv_counts = torch.zeros_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        send_split = send_counts.cpu().tolist()
        recv_split = recv_counts.cpu().tolist()

        # Bytes that actually traverse the network (exclude self-to-self)
        bytes_per_token = HIDDEN * 2  # bf16
        dispatch_bytes_local = sum(s for r, s in enumerate(send_split) if r != self.rank) * bytes_per_token
        comm_bytes["dispatch"] += dispatch_bytes_local

        # Build send buffer (token hidden states in send order)
        send_hid = x.index_select(0, sorted_tok)  # [N, HIDDEN]

        # ---- DISPATCH all-to-all ----
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        recv_hid = all_to_all_single_autograd(send_hid, send_split, recv_split)
        torch.cuda.synchronize()
        timings["dispatch"] += time.perf_counter() - t0

        # Also need to know which local expert each received token goes to.
        # We send the local-expert index alongside (small int tensor; comm-only,
        # not counted as MoE bandwidth since it's metadata).
        recv_local_exp = torch.empty(sum(recv_split), dtype=sorted_local_exp.dtype, device=device)
        dist.all_to_all_single(
            recv_local_exp, sorted_local_exp.to(torch.int64),
            output_split_sizes=recv_split, input_split_sizes=send_split,
        )

        # ---- LOCAL EXPERT COMPUTE ----
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_hid = torch.zeros_like(recv_hid)
        for e in range(self.local_n):
            mask = recv_local_exp == e
            if mask.any():
                out_hid[mask] = self.local_experts[e](recv_hid[mask])
        torch.cuda.synchronize()
        timings["compute"] += time.perf_counter() - t0

        # ---- COMBINE all-to-all (reverse) ----
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        recv_back = all_to_all_single_autograd(out_hid, recv_split, send_split)
        torch.cuda.synchronize()
        timings["combine"] += time.perf_counter() - t0

        combine_bytes_local = sum(s for r, s in enumerate(recv_split) if r != self.rank) * bytes_per_token
        comm_bytes["combine"] += combine_bytes_local

        # Weight & unsort
        recv_back = recv_back * sorted_w.unsqueeze(-1)
        unsort = torch.argsort(order)
        recv_back = recv_back.index_select(0, unsort)  # back to (token, expert) pair order
        recv_back = recv_back.view(T, num_active, HIDDEN)
        output = recv_back.sum(dim=1)
        return output

    # ---------------- forward ----------------
    def forward(self, x: torch.Tensor, timings: dict, comm_bytes: dict):
        # x: [T, HIDDEN]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        logits = self.gate(x)
        rw, ri = self.select_routed(logits)
        torch.cuda.synchronize()
        timings["route"] += time.perf_counter() - t0

        routed_out = self.dispatch_compute_combine(x, rw, ri, timings, comm_bytes)

        if self.condenser is not None:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            cw = F.softmax(self.cond_gate(x), dim=-1, dtype=torch.float32).to(x.dtype)
            cond_out = torch.zeros_like(x)
            for i, e in enumerate(self.condenser):
                cond_out = cond_out + e(x) * cw[:, i:i+1]
            torch.cuda.synchronize()
            timings["condenser"] += time.perf_counter() - t0
            return routed_out + cond_out
        return routed_out


# ---------------------------------------------------------------------------- #
# Main timing loop                                                              #
# ---------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=list(MODES), required=True)
    p.add_argument("--tokens_per_rank", type=int, default=4096,
                   help="Tokens generated per rank per step. Total batch = world_size * tokens_per_rank.")
    p.add_argument("--num_warmup", type=int, default=5)
    p.add_argument("--num_steps", type=int, default=20)
    p.add_argument("--output_dir", default="results_ep")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    # ---- Distributed init ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda")
    dtype = torch.bfloat16

    if rank == 0:
        print(f"[{args.mode}] world_size={world_size} tokens_per_rank={args.tokens_per_rank} "
              f"warmup={args.num_warmup} steps={args.num_steps}", flush=True)

    torch.manual_seed(args.seed + rank)

    layer = EPMoELayer(args.mode, dtype=dtype).to(device)
    if rank == 0:
        n_params = sum(p.numel() for p in layer.parameters())
        local_routed = sum(p.numel() for e in layer.local_experts for p in e.parameters())
        cond_params = sum(p.numel() for e in (layer.condenser or []) for p in e.parameters())
        print(f"[{args.mode}] params per rank: total={n_params/1e6:.1f}M  "
              f"local_routed={local_routed/1e6:.1f}M  replicated_condenser={cond_params/1e6:.1f}M",
              flush=True)

    # Synthetic input per rank — different across ranks, same across steps.
    x = torch.randn(args.tokens_per_rank, HIDDEN, dtype=dtype, device=device, requires_grad=False)

    timing_keys = ["route", "dispatch", "compute", "combine", "condenser", "total_fwd", "total_bwd"]
    comm_keys = ["dispatch", "combine"]

    def zero_dict(keys): return {k: 0.0 for k in keys}

    # ---- Warmup ----
    for i in range(args.num_warmup):
        ts = zero_dict(timing_keys); bs = zero_dict(comm_keys)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = layer(x, ts, bs)
        loss = out.float().pow(2).sum()
        torch.cuda.synchronize()
        ts["total_fwd"] = time.perf_counter() - t0
        torch.cuda.synchronize(); t1 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        ts["total_bwd"] = time.perf_counter() - t1
        layer.zero_grad(set_to_none=True)
        if rank == 0:
            print(f"[{args.mode}] warmup {i+1}/{args.num_warmup}  "
                  f"fwd={ts['total_fwd']*1000:6.2f}ms  bwd={ts['total_bwd']*1000:6.2f}ms", flush=True)

    # ---- Timed steps ----
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    all_steps = []  # list of per-step dicts
    for step in range(args.num_steps):
        ts = zero_dict(timing_keys); bs = zero_dict(comm_keys)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = layer(x, ts, bs)
        loss = out.float().pow(2).sum()
        torch.cuda.synchronize()
        ts["total_fwd"] = time.perf_counter() - t0

        torch.cuda.synchronize(); t1 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        ts["total_bwd"] = time.perf_counter() - t1

        layer.zero_grad(set_to_none=True)

        # Reduce timings & bytes across ranks: take MAX (slowest rank dominates)
        # and SUM for bytes (per-step network traffic across the whole job).
        ts_tensor = torch.tensor([ts[k] for k in timing_keys], device=device, dtype=torch.float64)
        bs_tensor = torch.tensor([bs[k] for k in comm_keys], device=device, dtype=torch.float64)
        dist.all_reduce(ts_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(bs_tensor, op=dist.ReduceOp.SUM)

        step_data = {k: ts_tensor[i].item() for i, k in enumerate(timing_keys)}
        for i, k in enumerate(comm_keys):
            step_data[f"bytes_{k}_total"] = bs_tensor[i].item()
        all_steps.append(step_data)

        if rank == 0:
            print(f"[{args.mode}] step {step+1:>2}/{args.num_steps}  "
                  f"fwd={step_data['total_fwd']*1000:6.1f}ms  "
                  f"bwd={step_data['total_bwd']*1000:6.1f}ms  "
                  f"disp={step_data['dispatch']*1000:5.1f}ms  "
                  f"comp={step_data['compute']*1000:5.1f}ms  "
                  f"comb={step_data['combine']*1000:5.1f}ms  "
                  f"cond={step_data['condenser']*1000:5.1f}ms  "
                  f"send_bytes={step_data['bytes_dispatch_total']/1e6:7.1f}MB",
                  flush=True)

    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    peak_mem_max_tensor = torch.tensor([peak_mem_gb], device=device, dtype=torch.float64)
    dist.all_reduce(peak_mem_max_tensor, op=dist.ReduceOp.MAX)
    peak_mem_gb_max = peak_mem_max_tensor.item()

    if rank == 0:
        def stats(key):
            vals = [s[key] for s in all_steps]
            return {
                "median_ms": statistics.median(vals) * 1000,
                "mean_ms":   statistics.mean(vals) * 1000,
                "std_ms":    (statistics.stdev(vals) if len(vals) > 1 else 0.0) * 1000,
            }
        def byte_stats(key):
            vals = [s[key] for s in all_steps]
            return {
                "median_bytes": statistics.median(vals),
                "mean_bytes":   statistics.mean(vals),
            }

        result = {
            "mode": args.mode,
            "world_size": world_size,
            "tokens_per_rank": args.tokens_per_rank,
            "total_tokens": args.tokens_per_rank * world_size,
            "num_steps": args.num_steps,
            "num_warmup": args.num_warmup,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "total_routed_experts": TOTAL_ROUTED_EXPERTS,
            "top_k": TOP_K,
            "num_condenser": NUM_CONDENSER if args.mode in ("ec_paper", "ec_code") else 0,
            "local_n_experts": TOTAL_ROUTED_EXPERTS // world_size,
            "peak_memory_gb_max_rank": peak_mem_gb_max,
            "route":     stats("route"),
            "dispatch":  stats("dispatch"),
            "compute":   stats("compute"),
            "combine":   stats("combine"),
            "condenser": stats("condenser"),
            "total_fwd": stats("total_fwd"),
            "total_bwd": stats("total_bwd"),
            "bytes_dispatch_total_per_step": byte_stats("bytes_dispatch_total"),
            "bytes_combine_total_per_step":  byte_stats("bytes_combine_total"),
            "all_steps": all_steps,
        }
        result["total_ms_median"] = result["total_fwd"]["median_ms"] + result["total_bwd"]["median_ms"]
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"{args.mode}_ep_results.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[{args.mode}] === EP RESULTS ===", flush=True)
        for k in ("route", "dispatch", "compute", "combine", "condenser", "total_fwd", "total_bwd"):
            print(f"[{args.mode}]   {k:<10} median {result[k]['median_ms']:7.2f} ms", flush=True)
        print(f"[{args.mode}]   total fwd+bwd median = {result['total_ms_median']:.2f} ms", flush=True)
        print(f"[{args.mode}]   dispatch bytes/step (all ranks): "
              f"{result['bytes_dispatch_total_per_step']['median_bytes']/1e6:.2f} MB", flush=True)
        print(f"[{args.mode}]   combine bytes/step (all ranks):  "
              f"{result['bytes_combine_total_per_step']['median_bytes']/1e6:.2f} MB", flush=True)
        print(f"[{args.mode}]   peak memory (max rank): {peak_mem_gb_max:.2f} GB", flush=True)
        print(f"[{args.mode}]   saved -> {out_path}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
