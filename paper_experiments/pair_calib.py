"""Coverage of CosmoMap-sigma's total uncertainty around CosmoMap's (MSE) residuals, i.e. the pairing
recommended in the paper. Per seed s: sigma from film_z0_s<s>, means from mse_z0_s<s>. The scale factor is
fitted on validation boxes (as in calib.py) when both runs have pred_val. Writes RUNS/pair_calibration.json.

    python pair_calib.py
"""
import json

import numpy as np
from scipy.special import erfinv
from scipy.stats import spearmanr

from common import RUNS, load_stats, prep_path, split, to_log_physical

z = 0.0
stats = load_stats()
levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.68, 0.8, 0.9, 0.95, 0.99])
zq = np.sqrt(2) * erfinv(levels)


def residuals(sub, folder, mean_run, sig_run, f, nsims=100):
    sims = split()[sub][:nsims]
    tr = np.load(prep_path(f, z), mmap_mode="r")
    mu = np.load(RUNS / mean_run / folder / f"mu_{f}_z{z:.1f}.npy", mmap_mode="r")
    sg = np.load(RUNS / sig_run / folder / f"sig_{f}_z{z:.1f}.npy", mmap_mode="r")
    r, s_ = [], []
    for k, s in enumerate(sims):
        t = to_log_physical(np.asarray(tr[s], np.float32), f, stats)
        r.append(np.abs(np.asarray(mu[k], np.float32) - t)[::4, ::4, ::4].ravel())
        s_.append(np.maximum(np.asarray(sg[k], np.float32), 1e-4)[::4, ::4, ::4].ravel())
    return np.concatenate(r), np.concatenate(s_)


def fit_scale(u):
    cs = np.linspace(0.3, 2.0, 341)
    err = [np.sum([((u < c * q).mean() - l) ** 2 for q, l in zip(zq, levels)]) for c in cs]
    return float(cs[int(np.argmin(err))])


out = {}
for seed in range(3):
    sig_run, mse_run = f"film_z0_s{seed}", f"mse_z0_s{seed}"
    has_val = (RUNS / sig_run / "pred_val").exists() and (RUNS / mse_run / "pred_val").exists()
    for f in ("T", "ne"):
        res = {}
        for label, mean_run in (("own", sig_run), ("paired", mse_run)):
            r, s = residuals("test", "pred", mean_run, sig_run, f)
            u = r / s
            d = dict(raw=[float((u < q).mean()) for q in zq], spearman=float(spearmanr(r[::50], s[::50])[0]))
            if has_val:
                rv, sv = residuals("val", "pred_val", mean_run, sig_run, f)
                c = fit_scale(rv / sv)
                d.update(scale=c, cal=[float((u < c * q).mean()) for q in zq])
            res[label] = d
            print(seed, f, label, "cov68 raw %.3f cov95 raw %.3f" % (d["raw"][6], d["raw"][9]),
                  ("scale %.3f cov68 cal %.3f cov95 cal %.3f" % (d["scale"], d["cal"][6], d["cal"][9])) if has_val else "(no val preds)",
                  "spearman(|res|,sig) %.2f" % d["spearman"], flush=True)
        out[f"s{seed}_{f}"] = res
out["levels"] = levels.tolist()
(RUNS / "pair_calibration.json").write_text(json.dumps(out, indent=1))
