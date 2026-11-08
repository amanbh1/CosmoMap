"""Citable parameter-recovery numbers: median and 16-84 per cent spread of predicted/true for each parameter
and feature source, from param_recovery.json (first forest seed, 100 test boxes, z=0)."""
import json

import numpy as np

from common import RUNS

r = json.load(open(RUNS / "param_recovery.json"))
yt = np.array(r["true_test_params"])
for src in ("truth", "dm", "nofilm", "film"):
    yp = np.array(r["pred_seed0"][src])
    for j, n in enumerate(r["param_names"][:3]):
        q = yp[:, j] / yt[:, j]
        lo, med, hi = np.percentile(q, [16, 50, 84])
        print(f"{src:7s} {n:8s} ratio median {med:.3f}  16-84 [{lo:.3f},{hi:.3f}]  half-width {(hi-lo)/2:.3f}  rms frac err {np.sqrt(np.mean((q-1)**2)):.3f}  R2 {r[src]['r2_mean'][j]:.2f}")
