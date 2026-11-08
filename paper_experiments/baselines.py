"""Non-neural baselines, scored on the same test set and saved in the same format
as predict.py (RUNS/<name>/pred/mu_<field>_z<z>.npy, log10 physical units).

B1 "binned":  per-redshift conditional mean of log10 target given log10 rho_DM
              (0.1-sigma bins, fitted on training sims). Equivalent to the best
              possible one-to-one mapping from local DM density.
B2 "gbt":     gradient-boosted trees on local multi-scale features:
              log rho_DM smoothed at sigma = 0,1,2,4,8 voxels, log|v| at 0,2,
              the six CAMELS parameters and z. Trained on ~2.5M voxels.

    python baselines.py
"""

from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import HistGradientBoostingRegressor

from common import FIELDS_OUT, RUNS, available_redshifts, load_params, load_stats, prep_path, split, to_log_physical

SIGMAS_RHO = [0, 1, 2, 4, 8]
SIGMAS_V = [0, 2]


def features(rho, v, params_std, z):
    fs = [rho if s == 0 else gaussian_filter(rho, s, mode="wrap") for s in SIGMAS_RHO]
    fs += [v if s == 0 else gaussian_filter(v, s, mode="wrap") for s in SIGMAS_V]
    F = np.stack([f.ravel() for f in fs], 1)
    glob = np.broadcast_to(np.concatenate([params_std, [z]])[None], (F.shape[0], 7))
    return np.concatenate([F, glob], 1).astype(np.float32)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--zs", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--nsim", type=int, default=160, help="training simulations used")
    ap.add_argument("--nvox", type=int, default=3000, help="voxels sampled per simulation and redshift for the trees")
    ap.add_argument("--iters", type=int, default=400, help="max boosting iterations (early stopping on 10%% held out)")
    a = ap.parse_args()
    stats = load_stats()
    zs = [float(z) for z in a.zs.split(",")] if a.zs else available_redshifts()
    sp = split()
    P = load_params()
    Pstd = (P - np.array(stats["param_mean"])) / np.array(stats["param_std"])
    rng = np.random.default_rng(0)
    arr = {(f, z): np.load(prep_path(f, z), mmap_mode="r") for f in ["Mcdm", "Vcdm", *FIELDS_OUT] for z in zs}

    # ---------- training data ----------
    NB = 180
    edges = np.linspace(-6, 12, NB + 1)
    sums = {(f, z): np.zeros(NB) for f in FIELDS_OUT for z in zs}
    cnts = {z: np.zeros(NB) for z in zs}
    Xs, Ys = [], {f: [] for f in FIELDS_OUT}
    tr_sub = rng.choice(sp["train"], a.nsim, replace=False)
    for z in zs:
        for sim in tr_sub:
            rho = np.asarray(arr[("Mcdm", z)][sim], np.float32)
            v = np.asarray(arr[("Vcdm", z)][sim], np.float32)
            b = np.clip(np.digitize(rho.ravel(), edges) - 1, 0, NB - 1)
            cnts[z] += np.bincount(b, minlength=NB)
            ys = {f: np.asarray(arr[(f, z)][sim], np.float32).ravel() for f in FIELDS_OUT}
            for f in FIELDS_OUT:
                sums[(f, z)] += np.bincount(b, weights=ys[f], minlength=NB)
            F = features(rho, v, Pstd[sim], z)
            idx = rng.choice(F.shape[0], a.nvox, replace=False)
            Xs.append(F[idx])
            for f in FIELDS_OUT:
                Ys[f].append(ys[f][idx])
        print("collected z", z, flush=True)
    X = np.concatenate(Xs)
    lut = {}
    for f in FIELDS_OUT:
        for z in zs:
            m = sums[(f, z)] / np.maximum(cnts[z], 1)
            filled = cnts[z] > 0
            lut[(f, z)] = np.interp(np.arange(NB), np.flatnonzero(filled), m[filled])
    gbt = {}
    for f in FIELDS_OUT:
        gbt[f] = HistGradientBoostingRegressor(max_iter=a.iters, learning_rate=0.1, max_leaf_nodes=127,
                                               early_stopping=True, random_state=0).fit(X, np.concatenate(Ys[f]))
        print("gbt fitted", f, gbt[f].n_iter_, flush=True)

    # ---------- predict test set ----------
    test = sp["test"]
    for name in ("binned", "gbt"):
        name = name + a.tag
        (RUNS / name / "pred").mkdir(parents=True, exist_ok=True)
        (RUNS / name / "config.json").write_text(json.dumps({"zs": zs, "no_cond": name.startswith("binned")}))
    for z in zs:
        outs = {(n, f): np.lib.format.open_memmap(RUNS / (n + a.tag) / "pred" / f"mu_{f}_z{z:.1f}.npy", "w+", np.float16,
                                                  (len(test), 128, 128, 128))
                for n in ("binned", "gbt") for f in FIELDS_OUT}
        for k, sim in enumerate(test):
            rho = np.asarray(arr[("Mcdm", z)][sim], np.float32)
            v = np.asarray(arr[("Vcdm", z)][sim], np.float32)
            b = np.clip(np.digitize(rho, edges) - 1, 0, NB - 1)
            F = features(rho, v, Pstd[sim], z)
            for f in FIELDS_OUT:
                outs[("binned", f)][k] = to_log_physical(lut[(f, z)][b], f, stats).astype(np.float16)
                outs[("gbt", f)][k] = to_log_physical(gbt[f].predict(F).reshape(rho.shape), f, stats).astype(np.float16)
        for o in outs.values():
            o.flush()
        print("predicted z", z, flush=True)


if __name__ == "__main__":
    main()
