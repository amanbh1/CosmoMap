"""Collect every number quoted in the paper from saved metrics, into paper_numbers.json + printout.

    python paper_numbers.py
"""
import json

import numpy as np

from common import BINNED, GBT, HETERO_SEEDS, MAIN_ALLZ, MAIN_SEEDS, NOFILM, RUNS, SMALL, load_stats, prep_path, split, to_log_physical

SEEDS = MAIN_SEEDS
out = {}


def per_run(run, z="0.0"):
    m = json.loads((RUNS / run / "metrics.json").read_text())
    s = m["summary"][z]
    recs = [r for r in m["records"] if f"{r['z']:.1f}" == z]
    row = {k: s[k][0] for k in ["T.r2", "T.rmse", "ne.r2", "ne.rmse", "eps.r2_log", "eps.proj_pearson_log",
                                 "eps.top1_T_r2", "eps.top1_T_bias", "eps.top1_ne_r2", "eps.top1_ne_bias"]}
    row["T.r2_boxsd"] = s["T.r2"][1]
    row["ne.r2_boxsd"] = s["ne.r2"][1]
    row["T.bias_boxsd"] = float(np.std([r["T"]["bias"] for r in recs]))
    row["log_Lratio_median"] = float(np.median(np.log10([r["eps"]["L_ratio"] for r in recs])))
    row["log_Lratio_sd"] = float(np.std(np.log10([r["eps"]["L_ratio"] for r in recs])))
    row["log_Tew_ratio_median"] = float(np.median(np.log10([r["eps"]["Tew_pred"] / r["eps"]["Tew_true"] for r in recs])))
    for f in ("T", "ne"):
        for reg in ["deep_voids", "voids", "sheets", "filaments", "outskirts", "cores"]:
            row[f"{f}.{reg}_r2"] = s[f"{f}.{reg}_r2"][0]
            row[f"{f}.{reg}_rmse"] = s[f"{f}.{reg}_rmse"][0]
        if f"{f}.cov68" in s:
            row[f"{f}.cov68"] = s[f"{f}.cov68"][0]
            row[f"{f}.cov95"] = s[f"{f}.cov95"][0]
    return row


for run in [BINNED, GBT, NOFILM, SMALL, "abl_nounc_z0"] + SEEDS + HETERO_SEEDS:
    out[run] = per_run(run)


def seedstats(runs):
    rows = [out[r] for r in runs]
    return {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1))] for k in rows[0]}


out["film_seeds"] = seedstats(SEEDS)          # main model (MSE loss)
out["hetero_seeds"] = seedstats(HETERO_SEEDS)  # likelihood-loss variant

# calibration
out["calibration"] = {r: json.loads((RUNS / r / "calibration.json").read_text()) for r in HETERO_SEEDS}

# redshift
out["redshift"] = {}
for run in [MAIN_ALLZ, "gbt_allz", "binned_allz"]:
    m = json.loads((RUNS / run / "metrics.json").read_text())["summary"]
    out["redshift"][run] = {z: {k: m[z][k][0] for k in ["T.r2", "T.rmse", "ne.r2", "ne.rmse", "eps.r2_log"]} for z in m}

# luminosity with validation-calibrated sigmas (seed 0, z=0)
stats = load_stats()
test = split()["test"]
cal = out["calibration"][HETERO_SEEDS[0]]
pred = RUNS / HETERO_SEEDS[0] / "pred"
mu = {f: np.load(pred / f"mu_{f}_z0.0.npy", mmap_mode="r") for f in ("T", "ne")}
sg = {f: np.load(pred / f"sig_{f}_z0.0.npy", mmap_mode="r") for f in ("T", "ne")}
tru = {f: np.load(prep_path(f, 0.0), mmap_mode="r") for f in ("T", "ne")}
rr = {"raw": [], "cal": [], "none": []}
for k, sim in enumerate(test):
    t = {f: to_log_physical(np.asarray(tru[f][sim], np.float32), f, stats) for f in ("T", "ne")}
    p = {f: np.asarray(mu[f][k], np.float64) for f in ("T", "ne")}
    s = {f: np.asarray(sg[f][k], np.float64) for f in ("T", "ne")}
    Lt = (10.0 ** (2 * t["ne"] + 0.5 * t["T"])).sum()
    base = 10.0 ** (2 * p["ne"] + 0.5 * p["T"])
    for key, cT, cn in (("none", 0, 0), ("raw", 1, 1), ("cal", cal["T"]["scale"], cal["ne"]["scale"])):
        s2 = 4 * (cn * s["ne"]) ** 2 + 0.25 * (cT * s["T"]) ** 2
        rr[key].append(np.log10((base * np.exp(0.5 * np.log(10) ** 2 * s2)).sum() / Lt))
out["luminosity_hetero_seed0"] = {k: {"median": float(np.median(v)), "sd": float(np.std(v))} for k, v in rr.items()}

(RUNS / "paper_numbers.json").write_text(json.dumps(out, indent=1))
fs = out["film_seeds"]
print("SEEDS mean+-sd:", {k: f"{v[0]:.3f}+-{v[1]:.3f}" for k, v in fs.items() if not any(x in k for x in ["voids", "sheets", "filaments", "outskirts"])})
print("cores/voids:", {k: round(v[0], 3) for k, v in fs.items() if any(x in k for x in ["deep_voids_", "cores_"])})
print("lum (hetero):", out["luminosity_hetero_seed0"])
print("HETERO seeds:", {k: f"{v[0]:.3f}+-{v[1]:.3f}" for k, v in out["hetero_seeds"].items() if k in ["T.r2", "T.rmse", "ne.r2", "ne.rmse", "eps.r2_log", "log_Lratio_median"]})
print("calib:", {r: {f: (round(c[f]["scale"], 2), round(c[f]["test_cal"][6], 3), round(c[f]["test_cal"][9], 3)) for f in ("T", "ne")} for r, c in out["calibration"].items()})
for run, d in out["redshift"].items():
    print(run, {z: (round(v["T.r2"], 3), round(v["T.rmse"], 3), round(v["ne.r2"], 3), round(v["ne.rmse"], 3)) for z, v in d.items()})
for run in [BINNED, GBT, NOFILM, SMALL, "abl_nounc_z0"] + SEEDS[:1] + HETERO_SEEDS[:1]:
    r = out[run]
    print(run, "Tbias_sd", round(r["T.bias_boxsd"], 3), "logL med", round(r["log_Lratio_median"], 2), "sd", round(r["log_Lratio_sd"], 2), "Tew", round(r["log_Tew_ratio_median"], 3))
