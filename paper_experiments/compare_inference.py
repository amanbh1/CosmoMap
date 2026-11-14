"""Inference-scheme comparison at z=0 (first seed): window size, core and margin of the tiling, and the
full-box periodic pass. Prints test R2 and the median log ratio of predicted to true box luminosity.

The first run of each model is its standard prediction (64^3 windows, core 32, margin 16: predict.py defaults).
The others are made from the same weights with, for example,

    python predict.py --name mse_z0_s0 --mc 0 --core 32 --margin 32 --as-run test_c32m32_mse_z0_s0
    python predict.py --name mse_z0_s0 --mc 0 --core 64 --margin 32 --as-run test_margin32
    python predict.py --name mse_z0_s0 --mc 0 --core 16 --margin 24 --as-run test_c16m24
    python predict.py --name mse_z0_s0 --mc 0 --periodic            --as-run test_periodic

followed by ``python metrics.py <run>``.
"""
import json

import numpy as np

from common import RUNS

ROWS = [
    ("CosmoMap         64^3 (core 32, margin 16)", "mse_z0_s0"),
    ("CosmoMap         64^3 (core 16, margin 24)", "test_c16m24"),
    ("CosmoMap         96^3 (core 32, margin 32)", "test_c32m32_mse_z0_s0"),
    ("CosmoMap        128^3 (core 64, margin 32)", "test_margin32"),
    ("CosmoMap        full box, periodic", "test_periodic"),
    ("No conditioning  64^3 (core 32, margin 16)", "nofilm_mse_z0_s0"),
    ("No conditioning  96^3 (core 32, margin 32)", "test_c32m32_nofilm_mse_z0_s0"),
    ("No conditioning full box, periodic", "test_periodic_nofilm"),
    ("CosmoMap-sigma   64^3 (core 32, margin 16)", "film_z0_s0"),
    ("CosmoMap-sigma   96^3 (core 32, margin 32)", "test_c32m32_film_z0_s0"),
    ("CosmoMap-sigma  128^3 (core 64, margin 32)", "test_c64m32_film_z0_s0"),
    ("CosmoMap-sigma  full box, periodic", "test_periodic_film_z0_s0"),
    ("Small network    64^3 (core 32, margin 16)", "small_mse_z0_s0"),
    ("Small network    96^3 (core 32, margin 32)", "test_c32m32_small_mse_z0_s0"),
    ("Small network   128^3 (core 64, margin 32)", "test_c64m32_small_mse_z0_s0"),
    ("Small network   full box, periodic", "test_periodic_small_mse_z0_s0"),
    ("Batch norm       64^3 (core 32, margin 16)", "bn_mse_z0_s0"),
    ("Batch norm       96^3 (core 32, margin 32)", "test_c32m32_bn_mse_z0_s0"),
    ("Batch norm      128^3 (core 64, margin 32)", "test_c64m32_bn_mse_z0_s0"),
    ("Batch norm      full box, periodic", "test_periodic_bn_mse_z0_s0"),
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
          f"logL med {np.median(lr):+.3f} sd {np.std(lr):.3f}  top1 ne bias {s['eps.top1_ne_bias'][0]:+.3f}")
