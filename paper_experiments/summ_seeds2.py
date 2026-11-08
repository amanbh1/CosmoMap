"""Seed means for the unconditioned network of every quantity the paper quotes from its first seed."""
import json
import numpy as np
from common import RUNS

KEYS = ["eps.r2_log", "T.cores_r2", "T.voids_r2", "T.sheets_r2", "ne.voids_bias", "ne.sheets_bias", "ne.cores_bias", "ne.bias"]
for name in ("mse", "nofilm_mse", "small_mse"):
    rows = []
    for s in range(3):
        m = json.load(open(RUNS / f"{name}_z0_s{s}" / "metrics.json"))
        S = m["summary"]["0.0"]
        recs = [r for r in m["records"] if f"{r['z']:.1f}" == "0.0"]
        rows.append([S[k][0] for k in KEYS] + [np.std([r["T"]["bias"] for r in recs])])
    rows = np.array(rows)
    print(name)
    for k, c in zip(KEYS + ["T.bias_boxsd"], rows.T):
        print(f"  {k:16s} {np.round(c, 4)} mean={c.mean():.4f} sd={c.std(ddof=1):.4f}")
