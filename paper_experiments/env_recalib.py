"""Environment-dependent recalibration of CosmoMap-sigma uncertainties (first seed, z=0).

Voxels are binned by the PREDICTED emissivity percentile within each box (no truth needed at test time):
[0,50), [50,80), [80,95), [95,99), [99,100]. One scale factor per field and bin is fitted on the 100 validation
boxes (least squares over nominal levels, as in calib.py) and applied to the test boxes. Coverage is reported
for all voxels and the truth-defined regimes of regime_calib.py, for the global and per-bin rescaling, with
CosmoMap-sigma's own means and with CosmoMap (MSE) means. Writes RUNS/env_recalibration.json."""
import json

import numpy as np
from scipy.special import erfinv

from common import RUNS, load_stats, prep_path, split, to_log_physical

z = 0.0
st = load_stats()
levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.68, 0.8, 0.9, 0.95, 0.99])
zq = np.sqrt(2) * erfinv(levels)
EDGES = [0, 50, 80, 95, 99, 100.01]
SIG = "film_z0_s0"
cal = json.load(open(RUNS / SIG / "calibration.json"))


def collect(sub, folder, mean_run):
    sims = split()[sub][:100]
    tr = {f: np.load(prep_path(f, z), mmap_mode="r") for f in ("T", "ne", "Mcdm")}
    mu = {f: np.load(RUNS / mean_run / folder / f"mu_{f}_z{z:.1f}.npy", mmap_mode="r") for f in ("T", "ne")}
    mus = {f: np.load(RUNS / SIG / folder / f"mu_{f}_z{z:.1f}.npy", mmap_mode="r") for f in ("T", "ne")}
    sg = {f: np.load(RUNS / SIG / folder / f"sig_{f}_z{z:.1f}.npy", mmap_mode="r") for f in ("T", "ne")}
    rows = {f: dict(u=[], b=[], dense=[], bright=[]) for f in ("T", "ne")}
    for k, s in enumerate(sims):
        t = {f: to_log_physical(np.asarray(tr[f][s], np.float32), f, st) for f in ("T", "ne")}
        rho = np.asarray(tr["Mcdm"][s], np.float32)
        let = 2 * t["ne"] + 0.5 * t["T"]
        lep = 2 * np.asarray(mus["ne"][k], np.float32) + 0.5 * np.asarray(mus["T"][k], np.float32)  # sigma model's own prediction
        pct = np.argsort(np.argsort(lep.ravel())).reshape(lep.shape) / lep.size * 100
        b = np.digitize(pct, EDGES) - 1
        dense, bright = rho >= np.percentile(rho, 95), let >= np.percentile(let, 99)
        for f in ("T", "ne"):
            u = np.abs(np.asarray(mu[f][k], np.float32) - t[f]) / np.maximum(np.asarray(sg[f][k], np.float32), 1e-4)
            # keep every voxel of the rare bins, subsample the bulk
            keep = (b >= 3) | ((np.arange(u.size).reshape(u.shape) % 8) == 0)
            for key, v in (("u", u), ("b", b), ("dense", dense), ("bright", bright)):
                rows[f][key].append(v[keep])
    return {f: {k: np.concatenate(v) for k, v in r.items()} for f, r in rows.items()}


def fit_scale(u):
    cs = np.linspace(0.2, 3.0, 561)
    err = [np.sum([((u < c * q).mean() - l) ** 2 for q, l in zip(zq, levels)]) for c in cs]
    return float(cs[int(np.argmin(err))])


out = {}
for label, mean_run in (("own", SIG), ("paired", "mse_z0_s0")):
    va, te = collect("val", "pred_val", mean_run), collect("test", "pred", mean_run)
    for f in ("T", "ne"):
        vg = va[f]
        g = fit_scale(vg["u"])
        sb = [fit_scale(vg["u"][vg["b"] == i]) for i in range(len(EDGES) - 1)]
        tt = te[f]
        cb = np.array(sb)[tt["b"]]
        res = dict(global_scale=g, bin_scales=sb)
        # the bulk was subsampled 1:8 for bins 0-2; weight voxels accordingly for 'all' coverage
        w = np.where(tt["b"] >= 3, 1.0, 8.0)
        for reg, m in (("all", np.ones_like(tt["dense"])), ("dense5", tt["dense"]), ("bright1", tt["bright"])):
            ww = w[m]
            cov = lambda c: [float(np.sum(ww * (tt["u"][m] < c * q)) / ww.sum()) for q in zq]
            res[reg] = dict(global_=cov(g), perbin=cov(cb[m]), raw=cov(1.0))
            print(f"{label:6s} {f:2s} {reg:7s} 68%: raw {res[reg]['raw'][6]:.3f} global {res[reg]['global_'][6]:.3f} per-bin {res[reg]['perbin'][6]:.3f}"
                  f" | 95%: raw {res[reg]['raw'][9]:.3f} global {res[reg]['global_'][9]:.3f} per-bin {res[reg]['perbin'][9]:.3f}", flush=True)
        print(f"{label:6s} {f:2s} scales global {g:.3f} per-bin " + " ".join(f"{x:.3f}" for x in sb), flush=True)
        out[f"{label}|{f}"] = res
out["levels"], out["edges"] = levels.tolist(), EDGES
(RUNS / "env_recalibration.json").write_text(json.dumps(out, indent=1))
