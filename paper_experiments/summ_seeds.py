"""Seed means and spreads for the ablation runs (CosmoMap, unconditioned, small network)."""
import json
import numpy as np
from common import RUNS

KEYS = ("T.r2", "T.rmse", "ne.r2", "ne.rmse", "eps.top1_ne_r2", "eps.top1_T_r2")
for name in ("mse", "nofilm_mse", "small_mse"):
    rows, Ls, sds = [], [], []
    for s in range(3):
        m = json.load(open(RUNS / f"{name}_z0_s{s}" / "metrics.json"))
        S = m["summary"]["0.0"]
        rows.append([S[k][0] if k in S else np.nan for k in KEYS])
        L = np.log10([r["eps"]["L_ratio"] for r in m["records"] if r.get("z", 0.0) in (0.0, "0.0")])
        Ls.append(np.median(L)); sds.append(np.std(L))
    rows = np.array(rows)
    print(name)
    for k, col in zip(KEYS, rows.T):
        print(f"  {k:16s} mean={np.mean(col):.4f} sd={np.std(col, ddof=1):.4f} vals={np.round(col, 4)}")
    print(f"  medlogL mean={np.mean(Ls):+.3f} sd={np.std(Ls, ddof=1):.3f} vals={np.round(Ls, 3)}")
    print(f"  L scatter mean={np.mean(sds):.3f} vals={np.round(sds, 3)}")
