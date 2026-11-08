"""Validation analyses on the z=0 test set:
  (1) scaling relations  n_e ∝ rho^alpha,  T ∝ rho^beta
  (2) bright-structure detection + surface-brightness beta-models
  (3) parameter-space robustness vs Mahalanobis distance

    python validation_extra.py
Writes RUNS/validation_extra.json
"""
import json

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit

from common import GBT, MAIN, NOFILM, RUNS, load_params, load_stats, prep_path, split, to_log_physical

Z = 0.0
MODELS = {"film": MAIN, "nofilm": NOFILM, "gbt": GBT}
STRUCT = np.ones((3, 3, 3), bool)
HALF = 20          # profile cube half-size in voxels (3.9 Mpc/h)
NR = 16            # radial bins out to HALF voxels
VOX = 25.0 / 128   # Mpc/h


def beta_model(r, logS0, rc, beta):
    return logS0 - (3 * beta - 0.5) * np.log10(1 + (r / rc) ** 2)


def fit_beta(r, prof):
    ok = prof > 0
    try:
        p, _ = curve_fit(beta_model, r[ok], np.log10(prof[ok]), p0=[np.log10(prof[ok][0]), 0.3, 0.7],
                         bounds=([-np.inf, 0.02, 0.2], [np.inf, 3.0, 3.0]), maxfev=20000)
        return p.tolist()
    except Exception:
        return [np.nan, np.nan, np.nan]


def cube(a, c):
    idx = [np.arange(ci - HALF, ci + HALF + 1) % 128 for ci in c]
    return a[np.ix_(*idx)]


R_EDGES = np.linspace(0, HALF, NR + 1)
R_MID = 0.5 * (R_EDGES[1:] + R_EDGES[:-1]) * VOX
yy, xx = np.mgrid[-HALF:HALF + 1, -HALF:HALF + 1]
RR = np.sqrt(xx ** 2 + yy ** 2)
RBIN = np.digitize(RR, R_EDGES) - 1


def profile(eps_cube):
    sb = eps_cube.sum(2)  # project along z
    return np.array([sb[RBIN == i].mean() for i in range(NR)])


def main():
    st = load_stats()
    sp = split()
    test = sp["test"]
    P = load_params()
    tr = {f: np.load(prep_path(f, Z), mmap_mode="r") for f in ("T", "ne", "Mcdm")}
    pr = {m: {f: np.load(RUNS / r / "pred" / f"mu_{f}_z{Z:.1f}.npy", mmap_mode="r") for f in ("T", "ne")} for m, r in MODELS.items()}
    rng = np.random.default_rng(0)

    # absolute detection threshold: 99.9th percentile of true log-emissivity over 30 validation boxes
    v = []
    for sim in sp["val"][:30]:
        lT = to_log_physical(np.asarray(tr["T"][sim], np.float32), "T", st)
        lne = to_log_physical(np.asarray(tr["ne"][sim], np.float32), "ne", st)
        v.append((2 * lne + 0.5 * lT).ravel()[::8])
    thr = float(np.percentile(np.concatenate(v), 99.9))

    scal = {s: {"alpha": [], "beta": [], "alpha_dense": [], "beta_dense": []} for s in ["truth", *MODELS]}
    det = {m: {"n_true": [], "n_pred": [], "matched": [], "Lratio_matched": []} for m in MODELS}
    prof = {s: [] for s in ["truth", *MODELS]}
    betas = {s: [] for s in ["truth", *MODELS]}

    for k, sim in enumerate(test):
        lrho = to_log_physical(np.asarray(tr["Mcdm"][sim], np.float32), "Mcdm", st)
        fields = {"truth": (to_log_physical(np.asarray(tr["T"][sim], np.float32), "T", st),
                            to_log_physical(np.asarray(tr["ne"][sim], np.float32), "ne", st))}
        for m in MODELS:
            fields[m] = (np.asarray(pr[m]["T"][k], np.float32), np.asarray(pr[m]["ne"][k], np.float32))

        # (1) scaling relations on 1e4 random voxels, and on the densest 5 per cent
        idx = rng.choice(lrho.size, 10000, replace=False)
        dense = np.flatnonzero(lrho.ravel() >= np.percentile(lrho, 95))
        idd = rng.choice(dense, min(10000, dense.size), replace=False)
        for s, (lT, lne) in fields.items():
            for key, ii in (("", idx), ("_dense", idd)):
                x = lrho.ravel()[ii]
                scal[s]["alpha" + key].append(float(np.polyfit(x, lne.ravel()[ii], 1)[0]))
                scal[s]["beta" + key].append(float(np.polyfit(x, lT.ravel()[ii], 1)[0]))

        # (2) detection: 26-connected components above the absolute threshold
        leps = {s: 2 * lne + 0.5 * lT for s, (lT, lne) in fields.items()}
        lab_t, n_t = ndimage.label(leps["truth"] > thr, structure=STRUCT)
        eps_t = 10.0 ** leps["truth"].astype(np.float64)
        if n_t:
            Lt = ndimage.sum(eps_t, lab_t, index=np.arange(1, n_t + 1))
        for m in MODELS:
            lab_p, n_p = ndimage.label(leps[m] > thr, structure=STRUCT)
            eps_p = 10.0 ** leps[m].astype(np.float64)
            det[m]["n_true"].append(int(n_t)); det[m]["n_pred"].append(int(n_p))
            if n_t:
                # a true structure is recovered if any predicted component overlaps it
                hit = ndimage.maximum(lab_p > 0, lab_t, index=np.arange(1, n_t + 1)).astype(bool)
                det[m]["matched"].append(float(hit.mean()))
                Lp = ndimage.sum(eps_p, lab_t, index=np.arange(1, n_t + 1))
                det[m]["Lratio_matched"] += np.log10(Lp[hit] / Lt[hit]).tolist()

        # (2b) surface-brightness profiles around the 3 brightest true peaks
        sm = ndimage.gaussian_filter(leps["truth"], 1.0, mode="wrap")
        peaks = np.argwhere((sm == ndimage.maximum_filter(sm, size=15, mode="wrap")) & (sm > thr))
        peaks = sorted(peaks, key=lambda c: -sm[tuple(c)])[:3]
        for c in peaks:
            for s in fields:
                p = profile(cube(10.0 ** leps[s].astype(np.float64), c))
                prof[s].append(p.tolist())
                betas[s].append(fit_beta(R_MID, p))
        if k % 20 == 0:
            print("box", k, flush=True)

    # (3) Mahalanobis distance of test parameters from the training distribution
    P = P.copy()
    P[:, 2:] = np.log(P[:, 2:])  # feedback parameters on a logarithmic scale
    Ptr = P[sp["train"]]
    C = np.cov(Ptr.T)
    d = Ptr.mean(0)
    dM = [float(np.sqrt((P[s] - d) @ np.linalg.solve(C, P[s] - d))) for s in test]
    m = json.loads((RUNS / MAIN / "metrics.json").read_text())
    recs = {r["sim"]: r for r in m["records"] if r["z"] == Z}
    rmseT = [recs[int(s)]["T"]["rmse"] for s in test]
    rmsene = [recs[int(s)]["ne"]["rmse"] for s in test]
    robust = dict(dM=dM, rmse_T=rmseT, rmse_ne=rmsene,
                  r_T=float(np.corrcoef(dM, rmseT)[0, 1]), r_ne=float(np.corrcoef(dM, rmsene)[0, 1]))

    out = dict(threshold_log_eps=thr, scaling=scal, detection=det, profiles=prof, beta_fits=betas,
               r_mid=R_MID.tolist(), robustness=robust)
    (RUNS / "validation_extra.json").write_text(json.dumps(out))

    print("SCALING (mean +- sd over boxes)")
    for s in scal:
        print(" ", s, {k2: f"{np.mean(v2):.3f}+-{np.std(v2):.3f}" for k2, v2 in scal[s].items()})
    print("DETECTION thr", round(thr, 2))
    for mname in det:
        D = det[mname]
        print(" ", mname, "n_true", np.sum(D["n_true"]), "n_pred", np.sum(D["n_pred"]), "recovered frac",
              round(np.mean(D["matched"]), 3), "median logL ratio (matched)", round(np.median(D["Lratio_matched"]), 3),
              "sd", round(np.std(D["Lratio_matched"]), 3))
    t = np.array(prof["truth"])
    for mname in MODELS:
        ratio = np.log10(np.array(prof[mname]) / t)
        b_t, b_p = np.array(betas["truth"]), np.array(betas[mname])
        ok = np.isfinite(b_t).all(1) & np.isfinite(b_p).all(1)
        print("PROFILE", mname, "median log ratio by radius", np.round(np.median(ratio, 0), 2).tolist())
        print("   beta true med", round(np.median(b_t[ok, 2]), 3), "pred med", round(np.median(b_p[ok, 2]), 3),
              "rc true med", round(np.median(b_t[ok, 1]), 3), "pred", round(np.median(b_p[ok, 1]), 3),
              "corr beta", round(np.corrcoef(b_t[ok, 2], b_p[ok, 2])[0, 1], 3), "corr rc", round(np.corrcoef(b_t[ok, 1], b_p[ok, 1])[0, 1], 3))
    print("ROBUSTNESS dM range", round(min(dM), 2), round(max(dM), 2), "mean", round(np.mean(dM), 2),
          "r(dM,rmseT)", round(robust["r_T"], 3), "r(dM,rmse_ne)", round(robust["r_ne"], 3))


if __name__ == "__main__":
    main()
