"""Read routing_stats.json (per-layer Gini) and emit r-allocation."""
import json
import os
import sys
import numpy as np

# Reuse water-fill from auto_allocate.py
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from auto_allocate import water_fill_allocate

STATS = os.path.join(HERE, "routing_stats.json")
data = json.load(open(STATS))
num_layers = max(int(k) for k in data) + 1
ginis = np.array([data[str(i)]["gini"] for i in range(num_layers)])

print(f"=== Path 2: routing-Gini proportional ===")
print(f"  num_layers={num_layers}  Gini range=[{ginis.min():.4f}, {ginis.max():.4f}]")
print()

# Allocate proportional to Gini directly
r_alloc = water_fill_allocate(ginis, total=48, max_per_layer=4, min_per_layer=0)

print(f"{'L':>3} {'gini':>8} {'r':>3}")
print("-" * 18)
for i in range(num_layers):
    print(f"{i:>3} {ginis[i]:>8.4f} {r_alloc[i]:>3d}")
print("-" * 18)
print(f"{'sum':>3} {ginis.sum():>8.4f} {r_alloc.sum():>3d}")
print()
print("PER_LAYER_R_GROUPS env value:")
print(",".join(str(int(x)) for x in r_alloc))
print()
print(f"Layer-group totals (early L0-7 / mid L8-15 / late L16-23):")
e, m, l = r_alloc[:8].sum(), r_alloc[8:16].sum(), r_alloc[16:24].sum()
print(f"  early: {e}   mid: {m}   late: {l}")
print()
print("Also try amplified (Gini^k) to force more differentiation:")
for k in (2, 3, 5):
    amp = ginis ** k
    r_amp = water_fill_allocate(amp, total=48, max_per_layer=4, min_per_layer=0)
    print(f"  gini^{k}: {r_amp.tolist()}  groupsum: e={r_amp[:8].sum()}/m={r_amp[8:16].sum()}/l={r_amp[16:24].sum()}")
