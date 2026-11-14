"""Paper figures.  python make_figures.py [fig ...]   (default: all available; figs_extra.py makes the rest)

Writes to ./figures/ (previews to ./fig_preview/).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
from scipy.special import erfinv  # noqa: E402

from common import BINNED, GBT, HETERO, MAIN, MAIN_ALLZ, NOFILM, RUNS, load_params, load_stats, prep_path, split, to_log_physical  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
PREVIEW = Path(__file__).resolve().parent / "fig_preview"
PREVIEW.mkdir(exist_ok=True)
_orig_savefig = matplotlib.figure.Figure.savefig


def _savefig(self, path, *a, **kw):
    _orig_savefig(self, path, *a, **kw)
    if str(path).endswith(".pdf"):
        _orig_savefig(self, PREVIEW / (Path(path).stem + ".png"), dpi=110)


matplotlib.figure.Figure.savefig = _savefig

# Categorical order fixed per model (validated palette; legends always shown).
MODELS = {
    "film": dict(label="CosmoMap", color="#2a78d6", marker="o", ls="-"),
    "gbt": dict(label="Gradient-boosted trees", color="#eb6834", marker="s", ls="--"),
    "nofilm": dict(label="U-Net, no conditioning", color="#1baf7a", marker="^", ls="-."),
    "binned": dict(label=r"Binned $\rho_{\rm DM}$ lookup", color="#eda100", marker="D", ls=":"),
}
RUN = {"film": MAIN, "gbt": GBT, "nofilm": NOFILM, "binned": BINNED}
FIELD_LABEL = {"T": r"$\log_{10} T$", "ne": r"$\log_{10} n_{\rm e}$", "eps": r"$\log_{10} \epsilon_X$"}
REG = ["deep_voids", "voids", "sheets", "filaments", "outskirts", "cores"]
REG_LABEL = ["Deep voids\n<P10", "Voids\nP10–30", "Sheets\nP30–50", "Filaments\nP50–80", "Outskirts\nP80–95", "Halo cores\n>P95"]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e4e0", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8a85",
    "xtick.color": "#55554f", "ytick.color": "#55554f", "lines.linewidth": 1.6,
    "savefig.dpi": 220, "savefig.bbox": "tight",
})


def metrics(run):
    p = RUNS / run / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def have(*runs):
    return all((RUNS / r / "metrics.json").exists() for r in runs)


# ------------------------------------------------------------------ slices
def fig_slices(z=0.0, sim_rank=0.5):
    stats = load_stats()
    test = split()["test"]
    m = metrics(RUN["film"])
    recs = [r for r in m["records"] if r["z"] == z]
    order = np.argsort([r["T"]["r2"] for r in recs])
    k = int(order[int(sim_rank * (len(order) - 1))])  # median-performing test box
    sim = test[k]
    sl = slice(60, 68)  # 8-voxel (1.56 Mpc/h) slab, averaged in log space

    def slab(a):
        return np.asarray(a, np.float32)[sl].mean(0)

    t = {f: slab(to_log_physical(np.asarray(np.load(prep_path(f, z), mmap_mode="r")[sim], np.float32), f, stats)) for f in ("T", "ne")}
    p = {f: slab(np.load(RUNS / RUN["film"] / "pred" / f"mu_{f}_z{z:.1f}.npy", mmap_mode="r")[k]) for f in ("T", "ne")}
    g = {f: slab(np.load(RUNS / RUN["gbt"] / "pred" / f"mu_{f}_z{z:.1f}.npy", mmap_mode="r")[k]) for f in ("T", "ne")}
    for d in (t, p, g):
        d["eps"] = 2 * d["ne"] + 0.5 * d["T"]
    rho = slab(to_log_physical(np.asarray(np.load(prep_path("Mcdm", z), mmap_mode="r")[sim], np.float32), "Mcdm", stats))
    vel = slab(to_log_physical(np.asarray(np.load(prep_path("Vcdm", z), mmap_mode="r")[sim], np.float32), "Vcdm", stats))

    fig, axes = plt.subplots(3, 6, figsize=(10.2, 5.5), constrained_layout=True)
    ext = [0, 25, 0, 25]
    for i, f in enumerate(["T", "ne", "eps"]):
        lo, hi = np.percentile(t[f], [1, 99.8])
        inp = [(rho, r"Input $\log_{10}\rho_{\rm DM}$"), (vel, r"Input $\log_{10}|v_{\rm DM}|$"), (None, "")][i]
        cols = [(inp[0], inp[1], "Greys", None),
                (t[f], "CAMELS-TNG (truth)", "magma", (lo, hi)),
                (p[f], "CosmoMap", "magma", (lo, hi)),
                (g[f], "Boosted trees", "magma", (lo, hi)),
                (p[f] - t[f], "CosmoMap $-$ truth", "RdBu_r", (-0.8, 0.8)),
                (g[f] - t[f], "Trees $-$ truth", "RdBu_r", (-0.8, 0.8))]
        for j, (img, title, cmap, lim) in enumerate(cols):
            ax = axes[i, j]
            ax.grid(False)
            if j == 0 and img is None:
                ax.axis("off")
                continue
            if j == 0:
                ax.set_title(title)
            im = ax.imshow(img.T, origin="lower", extent=ext, cmap=cmap,
                           vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1])
            ax.set_xticks([0, 10, 20]); ax.set_yticks([0, 10, 20])
            if i == 0:
                ax.set_title(title)
            if j == 1:
                ax.set_ylabel(FIELD_LABEL[f] + "\n" + r"$y\ [h^{-1}{\rm Mpc}]$")
            if i == 2:
                ax.set_xlabel(r"$x\ [h^{-1}{\rm Mpc}]$")
            if j in (0, 3, 5):
                cb = fig.colorbar(im, ax=ax, shrink=0.8)
                if j == 5:
                    cb.set_label("dex")
    fig.savefig(OUT / "slices_z0.png")
    plt.close(fig)
    print("slices: sim", sim, "T r2", recs[k]["T"]["r2"])


# ------------------------------------------------------------------ 2D histograms
def fig_hist2d(z=0.0, nsims=20):
    stats = load_stats()
    test = split()["test"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), constrained_layout=True)
    for i, f in enumerate(["T", "ne"]):
        tr = np.load(prep_path(f, z), mmap_mode="r")
        t = np.concatenate([to_log_physical(np.asarray(tr[s], np.float32), f, stats)[::2, ::2, ::2].ravel() for s in test[:nsims]])
        lo, hi = np.percentile(t, [0.05, 99.95])
        for j, mk in enumerate(["film", "gbt"]):
            pr = np.load(RUNS / RUN[mk] / "pred" / f"mu_{f}_z{z:.1f}.npy", mmap_mode="r")
            p = np.concatenate([np.asarray(pr[k], np.float32)[::2, ::2, ::2].ravel() for k in range(nsims)])
            ax = axes[i, j]
            ax.grid(False)
            h = ax.hist2d(t, p, bins=160, range=[[lo, hi], [lo, hi]], cmap="Blues", norm=matplotlib.colors.LogNorm())
            ax.plot([lo, hi], [lo, hi], color="#55554f", lw=1, ls="--")
            r2 = 1 - np.mean((t - p) ** 2) / np.var(t)
            ax.text(0.04, 0.94, f"$R^2$ = {r2:.3f}\nRMSE = {np.sqrt(np.mean((t - p) ** 2)):.3f} dex",
                    transform=ax.transAxes, va="top", fontsize=8)
            ax.set_title(f"{MODELS[mk]['label']}: {FIELD_LABEL[f]}")
            ax.set_xlabel("Truth"); ax.set_ylabel("Prediction")
        fig.colorbar(h[3], ax=axes[i, :], shrink=0.85, label="Voxels")
    fig.savefig(OUT / "hist2d_z0.png")
    plt.close(fig)


# ------------------------------------------------------------------ environment
def fig_regimes():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True, sharey=True)
    x = np.arange(len(REG))
    keys = [k for k in ["film", "nofilm", "gbt"] if have(RUN[k])]
    w = 0.8 / len(keys)
    for ax, f in zip(axes, ["T", "ne"]):
        for i, mk in enumerate(keys):
            s = metrics(RUN[mk])["summary"]["0.0"]
            vals = [s[f"{f}.{r}_r2"][0] for r in REG]
            errs = [s[f"{f}.{r}_r2"][1] for r in REG]
            ax.bar(x + (i - (len(keys) - 1) / 2) * w, vals, w * 0.92, yerr=errs, capsize=1.5,
                   color=MODELS[mk]["color"], label=MODELS[mk]["label"], error_kw=dict(lw=0.7, ecolor="#55554f"))
        ax.axhline(0, color="#8a8a85", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(REG_LABEL, fontsize=7.5)
        ax.set_title(f"{FIELD_LABEL[f]}: $R^2$ within each density regime")
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("$R^2$")
    axes[0].legend(loc="upper left", frameon=False, ncol=3)
    fig.savefig(OUT / "regimes_z0.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ power spectra
def fig_pk():
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), constrained_layout=True, sharex=True)
    for j, f in enumerate(["T", "ne", "eps"]):
        for mk in MODELS:
            m = metrics(RUN[mk])
            if m is None:
                continue
            k = np.array(m["k_centres"])
            s = m["summary"]["0.0"]
            st = MODELS[mk]
            axes[0, j].plot(k, s[f"{f}.rk"][0], color=st["color"], ls=st["ls"], label=st["label"])
            axes[1, j].plot(k, s[f"{f}.Tk"][0], color=st["color"], ls=st["ls"])
        axes[0, j].set_title(FIELD_LABEL[f].replace(r"\log_{10} ", ""))
        axes[0, j].set_ylim(0, 1.02)
        axes[1, j].axhline(1, color="#8a8a85", lw=0.8)
        axes[1, j].set_ylim(0, 2)
        axes[1, j].set_xscale("log")
        axes[1, j].set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    axes[0, 0].set_ylabel(r"Cross-correlation $r(k)$")
    axes[1, 0].set_ylabel(r"Transfer $\sqrt{P_{\rm pred}/P_{\rm true}}$")
    axes[0, 0].legend(frameon=False, loc="lower left")
    fig.savefig(OUT / "power_z0.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ calibration
def fig_calibration(z=0.0, nsims=100, run=None, cal=None):
    run = run or HETERO
    stats = load_stats()
    test = split()["test"]
    levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.68, 0.8, 0.9, 0.95, 0.99])
    zq = np.sqrt(2) * erfinv(levels)
    fig, ax = plt.subplots(figsize=(3.6, 3.4), constrained_layout=True)
    styles = {"T": ("#2a78d6", "o", "-"), "ne": ("#eb6834", "s", "--")}
    for f in ["T", "ne"]:
        tr = np.load(prep_path(f, z), mmap_mode="r")
        mu = np.load(RUNS / run / "pred" / f"mu_{f}_z{z:.1f}.npy", mmap_mode="r")
        sg = np.load(RUNS / run / "pred" / f"sig_{f}_z{z:.1f}.npy", mmap_mode="r")
        u = []
        for k in range(nsims):
            t = to_log_physical(np.asarray(tr[test[k]], np.float32), f, stats)
            u.append((np.abs(np.asarray(mu[k], np.float32) - t) / np.maximum(np.asarray(sg[k], np.float32), 1e-4))[::4, ::4, ::4].ravel())
        u = np.concatenate(u)
        c, mk, ls = styles[f]
        ax.plot(levels, [(u < q).mean() for q in zq], color=c, marker=mk, ms=4, ls=ls, label=FIELD_LABEL[f] + " (raw)")
        if cal is not None:
            ax.plot(levels, [(u < q * cal[f]).mean() for q in zq], color=c, marker=mk, ms=4, ls=":", mfc="white",
                    label=FIELD_LABEL[f] + " (recalibrated)")
    ax.plot([0, 1], [0, 1], color="#8a8a85", lw=0.8)
    ax.set_xlabel("Nominal interval"); ax.set_ylabel("Empirical coverage")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(OUT / "calibration_z0.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ parameter dependence
def fig_params(run=None):
    run = run or RUN["film"]
    m = metrics(run)
    recs = [r for r in m["records"] if r["z"] == 0.0]
    P = np.array([r["params"] for r in recs])
    names = [r"$\Omega_{\rm m}$", r"$\sigma_8$", r"$A_{\rm SN1}$", r"$A_{\rm AGN1}$", r"$A_{\rm SN2}$", r"$A_{\rm AGN2}$"]
    logx = [False, False, True, True, True, True]
    fig, axes = plt.subplots(2, 6, figsize=(11, 3.8), constrained_layout=True, sharey="row")
    for i, f in enumerate(["T", "ne"]):
        y = np.array([r[f]["rmse"] for r in recs])
        for j in range(6):
            ax = axes[i, j]
            ax.scatter(P[:, j], y, s=9, color=MODELS["film"]["color"], edgecolor="white", linewidth=0.4)
            rho = np.corrcoef(np.log(P[:, j]) if logx[j] else P[:, j], y)[0, 1]
            ax.text(0.05, 0.92, f"r = {rho:+.2f}", transform=ax.transAxes, fontsize=7.5, va="top")
            if logx[j]:
                ax.set_xscale("log")
                ax.set_xticks([0.25, 0.5, 1, 2, 4] if j in (2, 3) else [0.5, 1, 2])
                ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if i == 1:
                ax.set_xlabel(names[j])
        axes[i, 0].set_ylabel(f"RMSE {FIELD_LABEL[f]} [dex]")
    fig.savefig(OUT / "params_z0.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ redshift
def fig_redshift():
    runs = {"film": MAIN_ALLZ, "gbt": "gbt_allz", "binned": "binned_allz"}
    if not have(*runs.values()):
        print("redshift: missing runs"); return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    for ax, f in zip(axes, ["T", "ne"]):
        for mk, rn in runs.items():
            s = metrics(rn)["summary"]
            zs = sorted(float(z) for z in s)
            st = MODELS[mk]
            ax.errorbar(zs, [s[f"{z:.1f}"][f"{f}.rmse"][0] for z in zs], yerr=[s[f"{z:.1f}"][f"{f}.rmse"][1] for z in zs],
                        color=st["color"], marker=st["marker"], ms=4, ls=st["ls"], capsize=2, label=st["label"])
        ax.set_xlabel("Redshift $z$"); ax.set_ylabel(f"RMSE {FIELD_LABEL[f]} [dex]")
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    fig.savefig(OUT / "redshift.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ luminosity
def fig_luminosity():
    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    bins = np.linspace(-2, 1, 41)
    for mk in ["film", "gbt", "nofilm"]:
        m = metrics(RUN[mk])
        if m is None:
            continue
        r = np.log10([x["eps"]["L_ratio"] for x in m["records"] if x["z"] == 0.0])
        st = MODELS[mk]
        ax.hist(r, bins=bins, histtype="step", lw=1.6, color=st["color"], ls=st["ls"],
                label=f"{st['label']} (median ${np.median(r):+.2f}$)")  # mathtext gives a true minus sign
    ax.axvline(0, color="#8a8a85", lw=0.8)
    ax.set_xlabel(r"$\log_{10}(L_{\rm pred}/L_{\rm true})$, box-integrated emissivity")
    ax.set_ylabel("Test boxes")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7)  # above the axes, clear of the bars
    fig.savefig(OUT / "luminosity_z0.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ training
def fig_training():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), constrained_layout=True)
    stats = load_stats()
    for mk in ["film", "nofilm"]:
        p = RUNS / RUN[mk] / "history.json"
        if not p.exists():
            continue
        h = json.loads(p.read_text())
        e = [r["epoch"] for r in h]
        st = MODELS[mk]
        axes[0].plot(e, [r["train"] for r in h], color=st["color"], ls="-", label=f"{st['label']} train")
        axes[0].plot(e, [r["val"] for r in h], color=st["color"], ls="--", label=f"{st['label']} val")
        axes[1].plot(e, [np.sqrt(r["val_mse_T"]) * stats["std"]["T"] for r in h], color=st["color"], ls="-", marker="o", ms=2.5, label=f"{st['label']}: T")
        axes[1].plot(e, [np.sqrt(r["val_mse_ne"]) * stats["std"]["ne"] for r in h], color=st["color"], ls="--", marker="s", ms=2.5, label=f"{st['label']}: $n_e$")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation RMSE [dex]")
    axes[0].legend(frameon=False, fontsize=7); axes[1].legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "training.pdf")
    plt.close(fig)


def fig_calibration_cal():
    c = json.loads((RUNS / HETERO / "calibration.json").read_text())
    fig_calibration(cal={"T": c["T"]["scale"], "ne": c["ne"]["scale"]})


FIGS = dict(slices=fig_slices, hist2d=fig_hist2d, regimes=fig_regimes, pk=fig_pk, calibration=fig_calibration_cal,
            params=fig_params, redshift=fig_redshift, luminosity=fig_luminosity, training=fig_training)

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        try:
            FIGS[name]()
            print("ok", name, flush=True)
        except Exception as exc:  # keep going; report
            print("FAILED", name, repr(exc), flush=True)
