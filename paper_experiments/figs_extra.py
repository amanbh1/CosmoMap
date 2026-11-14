"""Figures for the validation analyses: parameter recovery (bars and scatter), scaling relations,
surface-brightness profiles, individual structures, bias by density regime, lightcones.

    python figs_extra.py [name ...]      (default: all)
"""
import json
import sys

import numpy as np
from scipy import ndimage

import make_figures as mf  # reuse style, palette and the PDF->PNG preview hook
from common import GBT, MAIN, NOFILM, RUNS, load_stats, prep_path, split, to_log_physical
from validation_extra import R_MID, beta_model, cube, fit_beta, profile

plt = mf.plt
OUT = mf.OUT
M = mf.MODELS
REG, REG_LABEL = mf.REG, mf.REG_LABEL
SRC = {
    "truth": dict(label="Simulated gas (truth)", color="#55554f", hatch=""),
    "dm": dict(label="Dark matter inputs only", color="#eda100", hatch=""),
    "nofilm": dict(label="U-Net, no conditioning", color=M["nofilm"]["color"], hatch=""),
    "film": dict(label="CosmoMap (given parameters)", color=M["film"]["color"], hatch="///"),
}
PNAMES = [r"$\Omega_{\rm m}$", r"$\sigma_8$", r"$A_{\rm SN1}$", r"$A_{\rm AGN1}$", r"$A_{\rm SN2}$", r"$A_{\rm AGN2}$"]


def fig_param_recovery():
    r = json.loads((RUNS / "param_recovery.json").read_text())
    fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
    x = np.arange(6)
    w = 0.2
    for i, s in enumerate(SRC):
        st = SRC[s]
        ax.bar(x + (i - 1.5) * w, r[s]["r2_mean"], w * 0.92, yerr=r[s]["r2_sd"], color=st["color"], hatch=st["hatch"],
               edgecolor="white", linewidth=0.5, label=st["label"], error_kw=dict(lw=0.7, ecolor="#55554f"))
    ax.axhline(0, color="#8a8a85", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(PNAMES)
    ax.set_ylabel("Test $R^2$ (random forest)")
    ax.set_ylim(-0.35, 1.05)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=7.5)
    fig.savefig(OUT / "param_recovery.pdf")
    plt.close(fig)


def fig_param_scatter():
    r = json.loads((RUNS / "param_recovery.json").read_text())
    yt = np.array(r["true_test_params"])
    names = PNAMES[:3]
    srcs = [("truth", "Simulated gas", "#55554f", "o"), ("nofilm", "U-Net, no conditioning", M["nofilm"]["color"], "^"),
            ("film", "CosmoMap (given parameters)", M["film"]["color"], "s")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), constrained_layout=True)
    for j in range(3):
        ax = axes[j]
        for s, lab, c, mk in srcs:
            yp = np.array(r["pred_seed0"][s])[:, j]
            ax.scatter(yt[:, j], yp, s=10, color=c, marker=mk, alpha=0.8, label=f"{lab} ($R^2$={r[s]['r2_mean'][j]:.2f})")
        lo, hi = yt[:, j].min(), yt[:, j].max()
        ax.plot([lo, hi], [lo, hi], color="#8a8a85", ls="--", lw=0.8)
        if j == 2:
            ax.set_xscale("log"); ax.set_yscale("log")
            import matplotlib.ticker as mt
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(mt.FixedLocator([0.25, 0.5, 1, 2, 4]))
                axis.set_major_formatter(mt.FormatStrFormatter("%g"))
                axis.set_minor_formatter(mt.NullFormatter())
        ax.set_xlabel(f"True {names[j]}"); ax.set_ylabel(f"Recovered {names[j]}")
        ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.savefig(OUT / "param_scatter.pdf")
    plt.close(fig)


def fig_scaling():
    v = json.loads((RUNS / "validation_extra.json").read_text())["scaling"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
    for ax, key, lab in zip(axes, ["alpha", "beta"], [r"$\alpha$ ($n_{\rm e}\propto\rho_{\rm DM}^{\alpha}$)", r"$\beta$ ($T\propto\rho_{\rm DM}^{\beta}$)"]):
        t = np.array(v["truth"][key])
        for mk in ["film", "gbt"]:
            st = M[mk]
            p = np.array(v[mk][key])
            ax.scatter(t, p, s=12, color=st["color"], marker=st["marker"], edgecolor="white", linewidth=0.4,
                       label=f"{st['label']} (mean {p.mean():.3f})")
        lo, hi = min(t.min(), *(np.array(v[m][key]).min() for m in ["film", "gbt"])), max(t.max(), *(np.array(v[m][key]).max() for m in ["film", "gbt"]))
        ax.plot([lo, hi], [lo, hi], color="#8a8a85", lw=0.8, ls="--")
        ax.set_xlabel(f"True {lab}"); ax.set_ylabel(f"Predicted {lab.split(' ')[0]}")
        ax.set_title(f"Truth mean {t.mean():.3f}", fontsize=8)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(OUT / "scaling.pdf")
    plt.close(fig)


def fig_profiles():
    v = json.loads((RUNS / "validation_extra.json").read_text())
    r = np.array(v["r_mid"])
    t = np.array(v["profiles"]["truth"])
    fig, axes = plt.subplots(2, 1, figsize=(4.4, 4.6), constrained_layout=True, sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    norm = np.median(t[:, 0])
    axes[0].plot(r, np.median(t, 0) / norm, color=SRC["truth"]["color"], lw=2, label="Simulation (truth)")
    for mk in ["film", "nofilm", "gbt"]:
        st = M[mk]
        p = np.array(v["profiles"][mk])
        axes[0].plot(r, np.median(p, 0) / norm, color=st["color"], ls=st["ls"], label=st["label"])
        q = np.log10(p / t)
        axes[1].plot(r, np.median(q, 0), color=st["color"], ls=st["ls"])
        axes[1].fill_between(r, *np.percentile(q, [16, 84], 0), color=st["color"], alpha=0.12, lw=0) if mk == "film" else None
    axes[0].set_yscale("log"); axes[0].set_ylabel("Median surface brightness\n(normalised)")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].axhline(0, color="#8a8a85", lw=0.8)
    axes[1].set_ylim(-1.0, 0.4)
    axes[1].set_ylabel(r"$\log_{10}(S_{\rm pred}/S_{\rm true})$")
    axes[1].set_xlabel(r"Projected radius $[h^{-1}{\rm Mpc}]$")
    fig.savefig(OUT / "sb_profiles.pdf")
    plt.close(fig)


def fig_clusters():
    """Three brightest structures of one test box: projected emissivity (truth, CosmoMap, residual) and profiles."""
    st = load_stats()
    test = split()["test"]
    v = json.loads((RUNS / "validation_extra.json").read_text())
    thr = v["threshold_log_eps"]
    # box with the most bright voxels among the first 20
    best, bestn = 0, -1
    for k in range(20):
        sim = test[k]
        lT = to_log_physical(np.asarray(np.load(prep_path("T", 0.0), mmap_mode="r")[sim], np.float32), "T", st)
        lne = to_log_physical(np.asarray(np.load(prep_path("ne", 0.0), mmap_mode="r")[sim], np.float32), "ne", st)
        n = int(((2 * lne + 0.5 * lT) > thr).sum())
        if n > bestn:
            best, bestn = k, n
    k = best; sim = test[k]
    lT = to_log_physical(np.asarray(np.load(prep_path("T", 0.0), mmap_mode="r")[sim], np.float32), "T", st)
    lne = to_log_physical(np.asarray(np.load(prep_path("ne", 0.0), mmap_mode="r")[sim], np.float32), "ne", st)
    pT = np.asarray(np.load(RUNS / MAIN / "pred" / "mu_T_z0.0.npy", mmap_mode="r")[k], np.float32)
    pne = np.asarray(np.load(RUNS / MAIN / "pred" / "mu_ne_z0.0.npy", mmap_mode="r")[k], np.float32)
    lt, lp = 2 * lne + 0.5 * lT, 2 * pne + 0.5 * pT
    sm = ndimage.gaussian_filter(lt, 1.0, mode="wrap")
    peaks = np.argwhere((sm == ndimage.maximum_filter(sm, size=15, mode="wrap")) & (sm > thr))
    peaks = sorted(peaks, key=lambda c: -sm[tuple(c)])[:3]
    fig, axes = plt.subplots(3, 4, figsize=(9.6, 7.0), constrained_layout=True)
    ext = [-20 * 25 / 128, 20 * 25 / 128] * 2
    for i, c in enumerate(peaks):
        et, ep = cube(10.0 ** lt.astype(np.float64), c), cube(10.0 ** lp.astype(np.float64), c)
        st_, sp_ = np.log10(et.sum(2)), np.log10(ep.sum(2))
        lo, hi = np.percentile(st_, [2, 99.9])
        for j, (img, title, cmap, lim) in enumerate([(st_, "Truth", "magma", (lo, hi)), (sp_, "CosmoMap", "magma", (lo, hi)),
                                                       (sp_ - st_, "Prediction − truth [dex]", "RdBu_r", (-1, 1))]):
            ax = axes[i, j]; ax.grid(False)
            im = ax.imshow(img.T, origin="lower", cmap=cmap, vmin=lim[0], vmax=lim[1], extent=ext)
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(f"Structure {i + 1}\n" + r"$[h^{-1}{\rm Mpc}]$")
            if j in (1, 2):
                fig.colorbar(im, ax=ax, shrink=0.8)
        ax = axes[i, 3]
        r = np.array(R_MID)
        pt, pp = profile(et), profile(ep)
        bt, bp = fit_beta(r, pt), fit_beta(r, pp)
        ax.plot(r, pt / pt[0], "o", ms=3.5, color="#55554f", label="Truth")
        ax.plot(r, pp / pt[0], "s", ms=3.5, color=M["film"]["color"], label="CosmoMap")
        rr = np.linspace(r[0], r[-1], 100)
        ax.plot(rr, 10 ** beta_model(rr, *bt) / pt[0], color="#55554f", lw=1, ls=":", label=fr"$\beta$-fit truth ($\beta$={bt[2]:.2f})")
        ax.plot(rr, 10 ** beta_model(rr, *bp) / pt[0], color=M["film"]["color"], lw=1, ls=":", label=fr"$\beta$-fit pred ($\beta$={bp[2]:.2f})")
        ax.set_yscale("log"); ax.legend(frameon=False, fontsize=6.5)
        if i == 0:
            ax.set_title("Surface brightness profile")
        if i == 2:
            ax.set_xlabel(r"$r\ [h^{-1}{\rm Mpc}]$")
    fig.savefig(OUT / "clusters.png")
    plt.close(fig)


def fig_bias():
    """Mean bias (prediction minus truth) within each density regime, with the scatter over test boxes."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), constrained_layout=True)
    x = np.arange(len(REG)); w = 0.26
    runs = {"film": MAIN, "nofilm": NOFILM, "gbt": GBT}
    for ax, f in zip(axes, ["T", "ne"]):
        for i, (mk, rn) in enumerate(runs.items()):
            s = json.loads((RUNS / rn / "metrics.json").read_text())["summary"]["0.0"]
            ax.bar(x + (i - 1) * w, [s[f"{f}.{r}_bias"][0] for r in REG], w * 0.92, yerr=[s[f"{f}.{r}_bias"][1] for r in REG],
                   color=M[mk]["color"], label=M[mk]["label"], capsize=1.5, error_kw=dict(lw=0.7, ecolor="#55554f"))
        ax.axhline(0, color="#8a8a85", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(REG_LABEL, fontsize=7.5)
        ax.set_title(f"{mf.FIELD_LABEL[f]}: mean bias (prediction − truth) [dex]")
    axes[0].legend(frameon=False, fontsize=7.5, ncol=3, loc="lower left")
    fig.savefig(OUT / "bias_regimes.pdf")
    plt.close(fig)


def fig_lightcone():
    d = np.load(RUNS / "lightcone_example.npz")
    L = json.loads((RUNS / "lightcone.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5), constrained_layout=True)
    lt, lp = np.log10(d["truth"]), np.log10(d["pred"])
    lo, hi = np.percentile(lt, [1, 99.9])
    for ax, img, title, cen in ((axes[0], lt, "Truth", d["cen_t"]), (axes[1], lp, "CosmoMap (multi-redshift)", d["cen_p"])):
        ax.grid(False)
        im = ax.imshow(img.T, origin="lower", cmap="magma", vmin=lo, vmax=hi, extent=[0, 60, 0, 60])
        if len(cen):
            ax.scatter(cen[:, 0] * 60 / 256, cen[:, 1] * 60 / 256, s=40, facecolor="none", edgecolor="#86b6ef", linewidth=0.9)
        ax.set_title(f"{title}: {len(cen)} sources"); ax.set_xlabel("arcmin"); ax.set_ylabel("arcmin")
    fig.colorbar(im, ax=axes[:2], shrink=0.85, label=r"$\log_{10}$ surface brightness (arb.)")
    ft = np.concatenate([np.array(r["flux_true"]) for r in L["lightcones"]])
    fp = np.concatenate([np.array(r["flux_pred"]) for r in L["lightcones"]])
    bins = np.logspace(np.log10(max(min(ft.min(), fp.min()), 1)), np.log10(max(ft.max(), fp.max())), 25)
    ax = axes[2]
    for f, lab, c, ls in ((ft, "Truth", "#55554f", "-"), (fp, "CosmoMap", M["film"]["color"], "--")):
        n = np.array([(f > b).sum() for b in bins]) / len(L["lightcones"])
        ok = n > 0
        ax.step(bins[ok], n[ok], where="post", color=c, ls=ls, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Source counts above local background"); ax.set_ylabel(r"$N(>S)$ per deg$^2$")
    ax.legend(frameon=False)
    fig.savefig(OUT / "lightcone.png")
    plt.close(fig)


FIGS = dict(param_recovery=fig_param_recovery, param_scatter=fig_param_scatter, scaling=fig_scaling, profiles=fig_profiles,
            clusters=fig_clusters, bias=fig_bias, lightcone=fig_lightcone)

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        try:
            FIGS[name](); print("ok", name, flush=True)
        except Exception as e:  # keep going; report
            print("FAILED", name, repr(e), flush=True)
