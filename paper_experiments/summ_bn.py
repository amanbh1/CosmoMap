"""Summarise the BatchNorm retrain against GroupNorm CosmoMap across inference schemes (Table B1 style)."""
import json
from common import RUNS
def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d: return None
        d = d[k]
    return d
runs = ["mse_z0_s0", "mse_z0_s1", "mse_z0_s2", "nofilm_mse_z0_s0", "nofilm_mse_z0_s1", "nofilm_mse_z0_s2", "small_mse_z0_s0", "small_mse_z0_s1", "small_mse_z0_s2"]
for r in runs:
    p = RUNS / r / "metrics.json"
    if not p.exists(): print(r, "missing"); continue
    m = json.load(open(p)); S = m["summary"]["0.0"]
    import numpy as np
    L = np.array([rec["eps"]["L_ratio"] for rec in m["records"] if rec.get("z", 0.0) in (0.0, "0.0")])
    print(f"{r:30s} n={len(L)} medlogL={np.median(np.log10(L)):+.3f} sd={np.std(np.log10(L)):.3f}", [round(S[k][0],4) for k in ("T.r2","ne.r2","eps.L_ratio","T.cores_r2","eps.top1_T_r2")])
