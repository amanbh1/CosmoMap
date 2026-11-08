"""Compare inference schemes: tile window size (GroupNorm statistics) vs context margin."""
import json

import numpy as np

from common import RUNS

ROWS = [
    ("CosmoMap  96^3 window, 64 core, margin 16", "test_margin16_ref"),
    ("CosmoMap  64^3 window, 16 core, margin 24", "test_c16m24"),
    ("CosmoMap  64^3 window, 32 core, margin 16", "test_c32m16"),
    ("CosmoMap  96^3 window, 32 core, margin 32", "test_c32m32_mse_z0_s0"),
    ("CosmoMap 112^3 window, 64 core, margin 24", "test_margin24"),
    ("CosmoMap 128^3 window, 64 core, margin 32", "test_margin32"),
    ("CosmoMap 128^3 full box, periodic", "mse_z0_s0"),
    ("Uncond.   96^3 window, 32 core, margin 32", "test_c32m32_nofilm_mse_z0_s0"),
    ("Uncond.  128^3 full box, periodic", "nofilm_mse_z0_s0"),
]
for label, r in ROWS:
    p = RUNS / r / "metrics.json"
    if not p.exists():
        print(f"{label:44s} (missing {r})")
        continue
    m = json.loads(p.read_text())
    recs = [x for x in m["records"] if x["z"] == 0.0]
    s = m["summary"]["0.0"]
    lr = np.log10([x["eps"]["L_ratio"] for x in recs])
    print(f"{label:44s} T R2 {s['T.r2'][0]:.3f}  ne R2 {s['ne.r2'][0]:.3f}  ne bias {s['ne.bias'][0]:+.3f}  "
          f"logL med {np.median(lr):+.3f}  top1 ne bias {s['eps.top1_ne_bias'][0]:+.3f}")
