"""Numbers quoted in the paper that the other scripts do not print themselves: the parameter count by
component, training convergence, the ablation table, per-regime and per-scale detail, the split between
aleatoric and epistemic variance, structure and profile statistics, gravity-only inputs, the tree-baseline
convergence checks, lightcone totals, and the Mahalanobis distance of prior draws.

Reads only saved runs and the json files written by the analysis scripts; a section whose inputs are
missing is skipped with a message.

    python quoted_numbers.py
"""
import json
from collections import defaultdict

import numpy as np

from common import GBT, HETERO, HETERO_SEEDS, MAIN, MAIN_ALLZ, MAIN_SEEDS, NOFILM, RUNS, load_params, split

REG = ["deep_voids", "voids", "sheets", "filaments", "outskirts", "cores"]
NOFILM_SEEDS = [f"nofilm_mse_z0_s{s}" for s in range(3)]
SMALL_SEEDS = [f"small_mse_z0_s{s}" for s in range(3)]


def load(run):
    return json.loads((RUNS / run / "metrics.json").read_text())


def z0(m):
    return m["summary"]["0.0"], [r for r in m["records"] if r["z"] == 0.0]


def log_lum(recs):
    return np.log10([r["eps"]["L_ratio"] for r in recs])


def section(title, fn):
    print(f"\n=== {title}")
    try:
        fn()
    except (FileNotFoundError, KeyError) as e:
        print(f"  skipped: {e!r}")


# ------------------------------------------------------------------ model size
def model_size():
    from model import build

    def group(name):
        if name.startswith("cond_dense"):
            return "Conditioning MLP"
        if "_film" in name:
            return "FiLM layers"
        for prefix, g in (("stem", "Input stem"), ("enc", "Encoder"), ("bottleneck", "Bottleneck"), ("dec", "Decoder")):
            if name.startswith(prefix):
                return g
        return "Output head" if name == "prediction" else "Other"

    m = build()
    groups = defaultdict(int)
    for layer in m.layers:
        if layer.count_params():
            groups[group(layer.name)] += layer.count_params()
    total = m.count_params()
    print("  CosmoMap total", total)
    for k, v in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"    {k:18s} {v:10d} {100 * v / total:5.1f}%")
    small = build(base=20, groups=4).count_params()
    print(f"  small network (base 20, 4 groups) {small}  ratio {total / small:.1f}")


# ------------------------------------------------------------------ training
def training():
    for r in MAIN_SEEDS + HETERO_SEEDS + NOFILM_SEEDS + SMALL_SEEDS + [MAIN_ALLZ]:
        p = RUNS / r / "history.json"
        if not p.exists():
            print(f"  {r:20s} (no history)")
            continue
        h = json.loads(p.read_text())
        v = np.array([x["val"] for x in h])
        print(f"  {r:20s} best epoch {int(np.argmin(v)) + 1:2d} of {len(v)}  change of val loss over last five epochs "
              f"{100 * abs(v[-1] - v[-5]) / v[-5]:.1f}%  hours {sum(x['sec'] for x in h) / 3600:.1f}")


# ------------------------------------------------------------------ ablation table
def ablation():
    def row(runs):
        out = defaultdict(list)
        for r in runs:
            s, recs = z0(load(r))
            for k in ("T.r2", "T.rmse", "ne.r2", "ne.rmse", "eps.top1_ne_r2", "eps.top1_ne_bias", "eps.top1_T_r2", "eps.top1_T_bias"):
                out[k].append(s[k][0])
            lr = log_lum(recs)
            out["logL"].append(float(np.median(lr))); out["logL_sd"].append(float(np.std(lr)))
        return out

    rows = [("CosmoMap (MSE), 3 seeds", MAIN_SEEDS), ("no conditioning, 3 seeds", NOFILM_SEEDS), ("small network, 3 seeds", SMALL_SEEDS),
            ("MSE + MAE + Huber", ["abl_nounc_z0"]), ("CosmoMap-sigma, 3 seeds", HETERO_SEEDS),
            ("pure Gaussian NLL", ["nll_z0_s0"]), ("beta-NLL, last epoch", ["betanll_z0_s0_last"]),
            ("beta-NLL, lowest val. loss", ["betanll_z0_s0"])]
    for label, runs in rows:
        if not all((RUNS / r / "metrics.json").exists() for r in runs):
            print(f"  {label:26s} (missing runs)")
            continue
        o = row(runs)
        sd = f" +- {np.std(o['logL'], ddof=1):.2f}" if len(runs) > 1 else ""
        print(f"  {label:26s} T {np.mean(o['T.r2']):.3f}/{np.mean(o['T.rmse']):.3f}  ne {np.mean(o['ne.r2']):.3f}/{np.mean(o['ne.rmse']):.3f}  "
              f"logL {np.mean(o['logL']):+.2f}{sd} (per seed {np.round(o['logL'], 2).tolist()}; box scatter {np.mean(o['logL_sd']):.2f})  "
              f"brightest 1%: ne R2 {np.mean(o['eps.top1_ne_r2']):.2f} bias {np.mean(o['eps.top1_ne_bias']):+.2f}, "
              f"T R2 {np.mean(o['eps.top1_T_r2']):.2f} bias {np.mean(o['eps.top1_T_bias']):+.2f}")


# ------------------------------------------------------------------ regimes, scales, parameters
def regimes():
    for f in ("T", "ne"):
        for key in ("r2", "rmse", "bias"):
            rows = {}
            for name, runs in (("CosmoMap(seeds)", MAIN_SEEDS), ("sigma(seeds)", HETERO_SEEDS), ("no cond", [NOFILM]), ("trees", [GBT])):
                rows[name] = np.round(np.mean([[z0(load(r))[0][f"{f}.{g}_{key}"][0] for g in REG] for r in runs], 0), 3).tolist()
            print(f"  {f} {key}: {rows}")


def scales():
    ms = [load(r) for r in MAIN_SEEDS]
    k = np.array(ms[0]["k_centres"])
    print("  k [h/Mpc]", np.round(k, 2).tolist())
    for f in ("T", "ne", "eps"):
        print(f"  CosmoMap {f} r(k)", np.round(np.mean([z0(m)[0][f"{f}.rk"][0] for m in ms], 0), 3).tolist())
        print(f"  CosmoMap {f} T(k)", np.round(np.mean([z0(m)[0][f"{f}.Tk"][0] for m in ms], 0), 2).tolist())
    g = z0(load(GBT))[0]
    for f in ("T", "ne", "eps"):
        print(f"  trees    {f} r(k)", np.round(g[f"{f}.rk"][0], 3).tolist())


def parameters():
    s, recs = z0(load(MAIN))
    P = np.array([r["params"] for r in recs])
    for f in ("T", "ne"):
        y = np.array([r[f]["rmse"] for r in recs])
        print(f"  corr(parameter, RMSE {f}) [log for feedback]:",
              np.round([np.corrcoef(np.log(P[:, j]) if j >= 2 else P[:, j], y)[0, 1] for j in range(6)], 2).tolist())
    print("  projected-emissivity Pearson r (mean, box sd):", np.round(s["eps.proj_pearson_log"], 3).tolist())
    print("  emission-weighted T, median log10(pred/true):",
          {r: round(float(np.median(np.log10([x["eps"]["Tew_pred"] / x["eps"]["Tew_true"] for x in z0(load(r))[1]]))), 3)
           for r in MAIN_SEEDS + [GBT]})


# ------------------------------------------------------------------ uncertainty
def variance_split():
    pred = RUNS / HETERO / "pred"
    for f in ("T", "ne"):
        sg = np.load(pred / f"sig_{f}_z0.0.npy", mmap_mode="r")
        se = np.load(pred / f"sige_{f}_z0.0.npy", mmap_mode="r")
        tot = np.concatenate([np.asarray(sg[k], np.float32)[::4, ::4, ::4].ravel() for k in range(len(sg))])
        epi = np.concatenate([np.asarray(se[k], np.float32)[::4, ::4, ::4].ravel() for k in range(len(se))])
        print(f"  {f}: MC-dropout share of the total variance {(epi ** 2).sum() / (tot ** 2).sum():.3f}  (mean total sigma {tot.mean():.3f} dex)")


def likelihood_calibration():
    for r in HETERO_SEEDS + ["nll_z0_s0", "betanll_z0_s0_last", "betanll_z0_s0"]:
        p = RUNS / r / "calibration.json"
        if not p.exists():
            print(f"  {r:14s} (no calibration.json)")
            continue
        c = json.loads(p.read_text())
        print(f"  {r:14s}", {f: dict(scale=round(c[f]["scale"], 3), raw68=round(c[f]["test_raw"][6], 3), raw95=round(c[f]["test_raw"][9], 3),
                                      cal68=round(c[f]["test_cal"][6], 3), cal95=round(c[f]["test_cal"][9], 3)) for f in ("T", "ne")})


# ------------------------------------------------------------------ structures, profiles, robustness
def structures():
    v = json.loads((RUNS / "validation_extra.json").read_text())
    for s, d in v["scaling"].items():
        print(f"  slopes {s:7s}", {k: f"{np.mean(x):.3f}+-{np.std(x):.3f}" for k, x in d.items()}, "| mean |slope - true| per box",
              {k: round(float(np.mean(np.abs(np.array(x) - np.array(v['scaling']['truth'][k])))), 3) for k, x in d.items()})
    for m, d in v["detection"].items():
        print(f"  structures {m:7s} n_true {sum(d['n_true'])} n_pred {sum(d['n_pred'])}  median log L ratio of recovered "
              f"{np.median(d['Lratio_matched']):+.2f}")
    t = np.array(v["profiles"]["truth"]); r = np.array(v["r_mid"])
    bt, bp = np.array(v["beta_fits"]["truth"]), np.array(v["beta_fits"]["film"])
    ok = np.isfinite(bt).all(1) & np.isfinite(bp).all(1)
    print(f"  profiles {len(t)}  fits ok {int(ok.sum())}  beta median truth {np.median(bt[ok, 2]):.3f} CosmoMap {np.median(bp[ok, 2]):.3f} "
          f"corr {np.corrcoef(bt[ok, 2], bp[ok, 2])[0, 1]:.2f}  fraction of true fits with r_c below one voxel {np.mean(bt[:, 1] < 25 / 128):.2f}")
    q = np.log10(np.array(v["profiles"]["film"]) / t)
    print("  radius [Mpc/h]            ", np.round(r, 2).tolist())
    print("  median log(pred/true)     ", np.round(np.median(q, 0), 2).tolist())
    print("  fraction of profiles below", np.round((q < 0).mean(0), 2).tolist())
    rb = v["robustness"]
    print(f"  Mahalanobis distance of test boxes {min(rb['dM']):.2f}-{max(rb['dM']):.2f}  corr with RMSE: T {rb['r_T']:+.2f} ne {rb['r_ne']:+.2f}")
    p = RUNS / "structure_purity.json"
    if p.exists():
        for m, a in json.loads(p.read_text()).items():
            print(f"  {m:7s} completeness {a['completeness_any']:.2f} (strict {a['completeness_half']:.2f})  purity {a['purity_any']:.2f} (strict {a['purity_half']:.2f})")


def prior_distance(n=200_000):
    P = load_params().astype(np.float64)
    P[:, 2:] = np.log(P[:, 2:])
    tr = P[split()["train"]]
    mu, C = tr.mean(0), np.cov(tr.T)
    rng = np.random.default_rng(0)
    lo = np.array([0.1, 0.6, np.log(0.25), np.log(0.25), np.log(0.5), np.log(0.5)])
    hi = np.array([0.5, 1.0, np.log(4.0), np.log(4.0), np.log(2.0), np.log(2.0)])
    d = rng.uniform(lo, hi, (n, 6)) - mu
    dm = np.sqrt(np.einsum("ij,jk,ik->i", d, np.linalg.inv(C), d))
    print(f"  prior draws: mean d_M {dm.mean():.2f}, 1st-99th percentile {np.percentile(dm, 1):.2f}-{np.percentile(dm, 99):.2f}")


# ------------------------------------------------------------------ inputs, baselines, lightcones
def gravity_only():
    for label, r in (("hydro inputs", MAIN), ("N-body density only", "test_nbody"), ("N-body density and velocity", "test_nbody_full")):
        s, recs = z0(load(r))
        lr = log_lum(recs)
        print(f"  {label:28s} T R2 {s['T.r2'][0]:.3f}  ne R2 {s['ne.r2'][0]:.3f}  logL median {np.median(lr):+.2f} sd {np.std(lr):.2f}")


def tree_convergence():
    for label, r in (("160 sims, z=0, 400 iterations", "gbt"), ("800 sims, z=0, 400 iterations", "gbt_full"),
                     ("800 sims, z=0, 2000 iterations", "gbt_full2k"), ("160 sims, all redshifts", "gbt_allz")):
        s = load(r)["summary"]["0.0"]
        print(f"  {label:32s} T R2 {s['T.r2'][0]:.4f}  ne R2 {s['ne.r2'][0]:.4f}")


def lightcones():
    d = json.loads((RUNS / "lightcone.json").read_text())
    L = d["lightcones"]
    nt, npred = sum(r["n_true"] for r in L), sum(r["n_pred"] for r in L)
    matched = sum(round(r["completeness"] * r["n_true"]) for r in L)
    real = sum(round(r["purity"] * r["n_pred"]) for r in L)
    print(f"  {len(L)} lightcones: true sources {nt}, predicted {npred}, recovered {matched} ({matched / nt:.2f}), real {real} ({real / npred:.2f})")
    print("  purity per lightcone", [round(r["purity"], 2) for r in L])
    print("  summary", {k: round(v, 3) for k, v in d["summary"].items()})


if __name__ == "__main__":
    for title, fn in (("Model size (Table 1)", model_size), ("Training", training), ("Ablation table", ablation),
                      ("Density regimes", regimes), ("Scale dependence", scales), ("Parameter dependence and projected maps", parameters),
                      ("Aleatoric and epistemic variance", variance_split), ("Likelihood variants: coverage", likelihood_calibration),
                      ("Scaling relations, structures, profiles", structures), ("Mahalanobis distance of prior draws", prior_distance),
                      ("Gravity-only inputs", gravity_only), ("Tree baseline convergence", tree_convergence), ("Lightcones", lightcones)):
        section(title, fn)
