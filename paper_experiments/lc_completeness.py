"""Lightcone source completeness and purity as a function of flux (from lightcone_v2.json): fraction of true
sources recovered above true-flux thresholds, and purity above predicted-flux thresholds. Fluxes are in
detection counts and are compared within each lightcone through its own flux percentiles and pooled."""
import json

import numpy as np

from common import RUNS

L = json.load(open(RUNS / "lightcone_v2.json"))["lightcones"]
ft, mt, fp, rp = [], [], [], []
for r in L:
    f = np.array(r["flux_true"]); m = np.zeros(len(f), bool); m[r["matched_true"]] = True
    ft.append(f); mt.append(m); fp.append(np.array(r["flux_pred"])); rp.append(np.array(r["pred_real"], bool))
ft, mt, fp, rp = map(np.concatenate, (ft, mt, fp, rp))
print("pooled true", len(ft), "recovered", mt.sum(), f"{mt.mean():.3f}", "| pred", len(fp), "real", rp.sum(), f"{rp.mean():.3f}")
print("true-flux percentiles:", np.percentile(ft, [10, 50, 90, 99]).round(0))
for q in (0, 50, 75, 90, 95, 99):
    thr = np.percentile(ft, q); s = ft >= thr
    print(f"true sources with flux >= P{q} ({thr:.0f} cts): n={s.sum():4d} completeness {mt[s].mean():.3f}")
for k in (1, 2, 4, 8, 16):
    thr = k * np.median(ft); s = ft >= thr
    if s.sum(): print(f"flux >= {k}x median true flux: n={s.sum():4d} completeness {mt[s].mean():.3f}")
for q in (0, 50, 90):
    thr = np.percentile(fp, q); s = fp >= thr
    print(f"pred sources with flux >= P{q}: n={s.sum():4d} purity {rp[s].mean():.3f}")
# brightest N true sources per lightcone
for N in (5, 10, 20):
    c = []
    for r in L:
        f = np.array(r["flux_true"]); m = np.zeros(len(f), bool); m[r["matched_true"]] = True
        idx = np.argsort(f)[::-1][:N]; c.append(m[idx].mean())
    print(f"brightest {N} true sources per lightcone: completeness {np.mean(c):.3f} (range {min(c):.2f}-{max(c):.2f})")
