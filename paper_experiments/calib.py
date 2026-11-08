"""Fit a per-field scale c for the predicted sigma on the VALIDATION split so that
empirical coverage matches nominal (least squares over levels 0.1..0.99), then
report test coverage before/after. Writes RUNS/<name>/calibration.json.

    python calib.py film_z0_s0 0.0      (needs predict.py --split val first)
"""
import json
import sys

import numpy as np
from scipy.special import erfinv

from common import RUNS, load_stats, prep_path, split, to_log_physical

name, z = sys.argv[1], float(sys.argv[2])
stats = load_stats()
levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.68, 0.8, 0.9, 0.95, 0.99])
zq = np.sqrt(2) * erfinv(levels)


def ratios(sub, folder, f, nsims=100):
    sims = split()[sub][:nsims]
    tr = np.load(prep_path(f, z), mmap_mode="r")
    mu = np.load(RUNS / name / folder / f"mu_{f}_z{z:.1f}.npy", mmap_mode="r")
    sg = np.load(RUNS / name / folder / f"sig_{f}_z{z:.1f}.npy", mmap_mode="r")
    u = []
    for k, s in enumerate(sims):
        t = to_log_physical(np.asarray(tr[s], np.float32), f, stats)
        u.append((np.abs(np.asarray(mu[k], np.float32) - t) / np.maximum(np.asarray(sg[k], np.float32), 1e-4))[::4, ::4, ::4].ravel())
    return np.concatenate(u)


out = {}
for f in ("T", "ne"):
    uv = ratios("val", "pred_val", f)
    cs = np.linspace(0.3, 1.5, 241)
    err = [np.sum([((uv < c * q).mean() - l) ** 2 for q, l in zip(zq, levels)]) for c in cs]
    c = float(cs[int(np.argmin(err))])
    ut = ratios("test", "pred", f)
    out[f] = dict(scale=c, test_raw=[float((ut < q).mean()) for q in zq], test_cal=[float((ut < c * q).mean()) for q in zq])
    print(f, "scale", round(c, 3), "test cov68 raw", round(out[f]["test_raw"][6], 3), "cal", round(out[f]["test_cal"][6], 3),
          "cov95 raw", round(out[f]["test_raw"][9], 3), "cal", round(out[f]["test_cal"][9], 3))
out["levels"] = levels.tolist()
(RUNS / name / "calibration.json").write_text(json.dumps(out, indent=1))
