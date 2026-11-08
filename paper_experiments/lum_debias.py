"""Test a log-normal mean correction for emissivity sums.

The network's mean prediction is (approximately) the conditional median of log10 eps,
so sums of eps (luminosities) are biased low. If log10 eps ~ N(mu, s^2) per voxel,
E[eps] = 10^mu * exp(0.5 * (ln10 * s)^2) with s^2 = 4 s_ne^2 + 0.25 s_T^2 (ignoring
the T-ne error correlation). We test this with the predicted sigmas scaled by a
factor c (c=1: raw sigmas; c<1 accounts for the over-dispersion seen in coverage).

    python lum_debias.py film_z0_s0 0.0
"""
import sys

import numpy as np

from common import RUNS, load_stats, prep_path, split, to_log_physical

name, z = sys.argv[1], float(sys.argv[2])
stats = load_stats()
test = split()["test"]
pred = RUNS / name / "pred"
mu = {f: np.load(pred / f"mu_{f}_z{z:.1f}.npy", mmap_mode="r") for f in ("T", "ne")}
sg = {f: np.load(pred / f"sig_{f}_z{z:.1f}.npy", mmap_mode="r") for f in ("T", "ne")}
tru = {f: np.load(prep_path(f, z), mmap_mode="r") for f in ("T", "ne")}
cs = [0.0, 0.5, 0.6, 0.7, 0.8, 1.0]
ratios = {c: [] for c in cs}
for k, sim in enumerate(test):
    t = {f: to_log_physical(np.asarray(tru[f][sim], np.float32), f, stats) for f in ("T", "ne")}
    p = {f: np.asarray(mu[f][k], np.float64) for f in ("T", "ne")}
    s2 = 4 * np.asarray(sg["ne"][k], np.float64) ** 2 + 0.25 * np.asarray(sg["T"][k], np.float64) ** 2
    Lt = (10.0 ** (2 * t["ne"] + 0.5 * t["T"])).sum()
    base = 10.0 ** (2 * p["ne"] + 0.5 * p["T"])
    for c in cs:
        ratios[c].append((base * np.exp(0.5 * np.log(10) ** 2 * c**2 * s2)).sum() / Lt)
for c in cs:
    r = np.array(ratios[c])
    print(f"c={c:.1f}  L_pred/L_true: mean {r.mean():.3f}  median {np.median(r):.3f}  "
          f"16-84% [{np.percentile(r, 16):.2f}, {np.percentile(r, 84):.2f}]  log10 scatter {np.std(np.log10(r)):.3f}")
