"""Surface brightness profile ratio pred/truth by radius (validation_extra.json profiles), for the outskirts comment."""
import json
import numpy as np
from common import RUNS
v = json.load(open(RUNS / "validation_extra.json"))
P = v["profiles"]
t = np.array(P["truth"]); print("keys", list(P), t.shape)
r = np.array(v["r_mid"])
for mk in [k for k in P if k not in ("truth", "r", "radii")]:
    p = np.array(P[mk])
    if p.shape != t.shape: continue
    lr = np.log10(np.maximum(p, 1e-300) / np.maximum(t, 1e-300))
    med = np.median(lr, 0); lo, hi = np.percentile(lr, [16, 84], 0)
    print(mk); [print(f"  r={rr:6.2f} med {m:+.3f} [{a:+.3f},{b:+.3f}]") for rr, m, a, b in zip(r, med, lo, hi)]
