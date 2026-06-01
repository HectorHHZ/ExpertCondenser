"""
Auto-allocate per-layer condenser count r_ell from Phase-1 bias JSON.

Rule A1: bias-spread proportional.
  Score per layer s_ell = max(bias_ell) - min(bias_ell).
  Allocate r_ell proportional to s_ell, then enforce constraints:
    sum(r) == TOTAL, 0 <= r_ell <= MAX_PER_LAYER, r_ell integer.

Usage:
    python auto_allocate.py [--bias_file PATH] [--total 48] [--max 4]
"""

import argparse
import json
import sys
import numpy as np


def load_per_layer_bias(bias_file: str):
    """Returns dict layer_idx -> np.array of bias values."""
    with open(bias_file) as f:
        data = json.load(f)
    states = data["moe_bias_states"]
    out = {}
    import re
    for module_name, info in states.items():
        m = re.search(r"layers\.(\d+)\.", module_name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        out[layer_idx] = np.array(info["bias_values"], dtype=np.float64)
    return out


def water_fill_allocate(scores, total, max_per_layer, min_per_layer=0):
    """Integer allocation: sum=total, min_per_layer <= r <= max_per_layer, r ~ scores.

    Uses iterative capping with largest-remainder rounding.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    if total > n * max_per_layer:
        raise ValueError(f"infeasible: total={total} > n*max={n*max_per_layer}")
    if total < n * min_per_layer:
        raise ValueError(f"infeasible: total={total} < n*min={n*min_per_layer}")

    # Continuous proportional allocation with iterative cap enforcement.
    w = np.clip(scores, 1e-12, None)
    capped = np.zeros(n, dtype=bool)
    r = np.zeros(n, dtype=np.float64)
    remaining_total = float(total) - n * min_per_layer
    r += min_per_layer  # apply floor

    while True:
        free = ~capped
        wr = w[free]
        if wr.sum() == 0:
            break
        add = remaining_total * wr / wr.sum()
        # which become capped after adding?
        proposed = r[free] + add
        over = proposed > max_per_layer
        if not over.any():
            r[free] = proposed
            break
        # cap the over-ones, redistribute the excess in next iteration
        free_idx = np.where(free)[0]
        cap_idx = free_idx[over]
        excess = (proposed[over] - max_per_layer).sum()
        r[cap_idx] = max_per_layer
        capped[cap_idx] = True
        # The other 'free' indices get nothing yet; loop continues to redistribute
        remaining_total = excess + (proposed[~over] - r[free_idx[~over]]).sum()
        # ^ note: we already counted what would have gone to the not-over free ones; reset them
        r[free_idx[~over]] = r[free_idx[~over]]  # unchanged, will be re-allocated next iter
        # Recompute remaining_total cleanly: total - sum of capped/floored
        remaining_total = float(total) - r[capped].sum() - r[~capped].sum()

    # Largest-remainder rounding while respecting cap.
    r_floor = np.floor(r).astype(int)
    remainder = total - r_floor.sum()
    frac = r - np.floor(r)
    # Order by fractional part desc; assign +1 to top `remainder` that aren't yet at cap.
    order = np.argsort(-frac)
    i = 0
    while remainder > 0 and i < n:
        idx = order[i]
        if r_floor[idx] < max_per_layer:
            r_floor[idx] += 1
            remainder -= 1
        i += 1
    # If still short, fall back to incrementing any uncapped layers in score order.
    if remainder > 0:
        for idx in np.argsort(-w):
            while remainder > 0 and r_floor[idx] < max_per_layer:
                r_floor[idx] += 1
                remainder -= 1
            if remainder == 0:
                break

    # If we somehow overshot (shouldn't happen), decrement lowest-scored layers.
    while r_floor.sum() > total:
        for idx in np.argsort(w):
            if r_floor[idx] > min_per_layer:
                r_floor[idx] -= 1
                if r_floor.sum() == total:
                    break

    assert r_floor.sum() == total, f"got {r_floor.sum()} != {total}"
    assert (r_floor <= max_per_layer).all()
    assert (r_floor >= min_per_layer).all()
    return r_floor


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bias_file",
        default="<PATH>",
    )
    p.add_argument("--total", type=int, default=48)
    p.add_argument("--max", type=int, default=4)
    p.add_argument("--min", type=int, default=0)
    args = p.parse_args()

    biases = load_per_layer_bias(args.bias_file)
    num_layers = max(biases.keys()) + 1
    # Path 1 (std-as-proxy): bias-spread is degenerate (saturated at +/-0.5).
    # std captures the bimodal balance — higher std = more even split = more
    # information in the saturation pattern.
    scores = np.array([biases[i].std() for i in range(num_layers)])

    r_alloc = water_fill_allocate(scores, args.total, args.max, args.min)

    print(f"=== Rule A1' (Path 1): bias-std proportional ===")
    print(f"  num_layers={num_layers}, total={args.total}, max={args.max}, min={args.min}")
    print()
    print(f"{'Layer':>5}  {'std(bias)':>10}  {'r_alloc':>7}")
    print("-" * 32)
    for i in range(num_layers):
        print(f"{i:>5}  {scores[i]:>10.4f}  {r_alloc[i]:>7d}")
    print("-" * 32)
    print(f"{'sum':>5}  {scores.sum():>10.4f}  {r_alloc.sum():>7d}")
    print()
    print("Comma-separated for PER_LAYER_R_GROUPS env var:")
    print(",".join(str(int(x)) for x in r_alloc))
    print()
    print(f"Layer-group summary (early L0-7 / mid L8-15 / late L16-23):")
    g_early = r_alloc[:8].sum()
    g_mid = r_alloc[8:16].sum()
    g_late = r_alloc[16:24].sum()
    print(f"  early total: {g_early}  (mean {g_early/8:.2f})")
    print(f"  mid total:   {g_mid}  (mean {g_mid/8:.2f})")
    print(f"  late total:  {g_late}  (mean {g_late/8:.2f})")


if __name__ == "__main__":
    main()
