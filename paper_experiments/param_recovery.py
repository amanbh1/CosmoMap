"""Cosmological/astrophysical parameter recovery from summary statistics.

A random forest is trained on 32 summary statistics of fields from the 100 VALIDATION boxes
and tested on the 100 TEST boxes (z=0), so no U-Net training box is involved.
Feature sources:
  truth   - simulated T, ne (upper bound)
  dm      - dark matter inputs only (what is available without a gas model)
  nofilm  - unconditioned U-Net predictions (fair test: no parameter information injected)
  film    - FiLM U-Net predictions (partly circular: the network is given the true parameters)

    python param_recovery.py
"""
import json

import numpy as np
from scipy import ndimage, stats as sst
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from common import BOX, MAIN, NOFILM, PARAM_NAMES, RUNS, load_params, load_stats, prep_path, split, to_log_physical

Z = 0.0
KTARGETS = [1.0, 2.0, 5.0, 10.0]  # h/Mpc
k1 = np.fft.fftfreq(128, d=BOX / 128) * 2 * np.pi
kz = np.fft.rfftfreq(128, d=BOX / 128) * 2 * np.pi
KK = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2)
KMASKS = [np.abs(np.log(np.maximum(KK, 1e-9) / k)) < 0.15 for k in KTARGETS]
STRUCT = np.ones((3, 3, 3), bool)  # 26-connectivity


def pk_at(logx):
    x = 10.0 ** logx.astype(np.float64)
    f = np.abs(np.fft.rfftn(x / x.mean() - 1)) ** 2
    return [float(np.log10(f[m].mean())) for m in KMASKS]


def moments(a):
    a = a.ravel()[::4]
    return [a.mean(), a.std(), sst.skew(a), sst.kurtosis(a), *np.percentile(a, [10, 50, 90])]


def morph(field, thr):
    lab, n = ndimage.label(field > thr, structure=STRUCT)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
    return [float(n), float((field > thr).mean()), float(sizes.mean())]


def gas_features(lT, lne, thr):
    leps = 2 * lne + 0.5 * lT
    f = moments(lT) + moments(lne) + moments(leps)            # 21
    f += pk_at(lne) + pk_at(leps)                              # 8
    f += morph(leps, thr["eps"])[:2] + [float((lT > 6.0).mean())]  # 3  -> 32
    return f


def dm_features(lrho, lv, thr):
    f = moments(lrho) + moments(lv)                            # 14
    f += pk_at(lrho)                                           # 4
    f += morph(lrho, thr["rho"])                               # 3  -> 21
    return f


def main():
    st = load_stats()
    sp = split()
    P = load_params()
    arr = {f: np.load(prep_path(f, Z), mmap_mode="r") for f in ("T", "ne", "Mcdm", "Vcdm")}

    def truth(f, sim):
        return to_log_physical(np.asarray(arr[f][sim], np.float32), f, st)

    # absolute thresholds from validation truth
    vals_eps, vals_rho = [], []
    for sim in sp["val"][:30]:
        vals_eps.append((2 * truth("ne", sim) + 0.5 * truth("T", sim)).ravel()[::16])
        vals_rho.append(truth("Mcdm", sim).ravel()[::16])
    thr = {"eps": float(np.percentile(np.concatenate(vals_eps), 99)), "rho": float(np.percentile(np.concatenate(vals_rho), 99))}

    def pred(run, folder, f, k):
        return np.asarray(np.load(RUNS / run / folder / f"mu_{f}_z{Z:.1f}.npy", mmap_mode="r")[k], np.float32)

    feats = {s: {"val": [], "test": []} for s in ("truth", "dm", "nofilm", "film")}
    for part, folder in (("val", "pred_val"), ("test", "pred")):
        for k, sim in enumerate(sp[part]):
            lT, lne = truth("T", sim), truth("ne", sim)
            feats["truth"][part].append(gas_features(lT, lne, thr))
            feats["dm"][part].append(dm_features(truth("Mcdm", sim), truth("Vcdm", sim), thr))
            for src, run in (("nofilm", NOFILM), ("film", MAIN)):
                feats[src][part].append(gas_features(pred(run, folder, "T", k), pred(run, folder, "ne", k), thr))
        print("features", part, flush=True)

    ytr, yte = P[sp["val"]], P[sp["test"]]
    mu, sd = ytr.mean(0), ytr.std(0)
    res = {}
    for src in feats:
        Xtr, Xte = np.array(feats[src]["val"]), np.array(feats[src]["test"])
        r2s, r2tr, biases = [], [], []
        for seed in range(5):  # average over forest seeds
            rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=8)
            rf.fit(Xtr, (ytr - mu) / sd)
            yp = rf.predict(Xte) * sd + mu
            ypt = rf.predict(Xtr) * sd + mu
            r2s.append([r2_score(yte[:, j], yp[:, j]) for j in range(6)])
            r2tr.append([r2_score(ytr[:, j], ypt[:, j]) for j in range(6)])
            rng_p = P.max(0) - P.min(0)
            biases.append(((yp - yte).mean(0) / rng_p).tolist())
            if seed == 0:
                res.setdefault("pred_seed0", {})[src] = yp.tolist()
        r2s = np.array(r2s)
        res[src] = {"r2_mean": r2s.mean(0).tolist(), "r2_sd": r2s.std(0).tolist(), "n_features": Xtr.shape[1],
                    "r2_train": np.mean(r2tr, 0).tolist(), "bias_frac_range": np.mean(biases, 0).tolist(),
                    "rel_err_median": np.median(np.abs(np.array(res["pred_seed0"][src]) - yte) / yte, 0).tolist()}
        print(src, Xtr.shape[1], {n: round(v, 3) for n, v in zip(PARAM_NAMES, r2s.mean(0))}, 'train', [round(v, 2) for v in res[src]['r2_train']], 'bias', [round(v, 3) for v in res[src]['bias_frac_range']], 'relerr', [round(v, 2) for v in res[src]['rel_err_median']], flush=True)
    res["true_test_params"] = yte.tolist()
    res["param_names"] = PARAM_NAMES
    (RUNS / "param_recovery.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
