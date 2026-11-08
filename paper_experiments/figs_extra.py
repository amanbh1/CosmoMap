"""Figures for the validation analyses: parameter recovery, scaling relations,
surface-brightness profiles, lightcones.   python figs_extra.py
"""
import json

import numpy as np

import make_figures as mf  # reuse style, palette and the PDF->PNG preview hook
from common import RUNS

plt = mf.plt
OUT = mf.OUT
M = mf.MODELS
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


if __name__ == "__main__":
    for f in (fig_param_recovery, fig_scaling, fig_profiles, fig_lightcone):
        try:
            f(); print("ok", f.__name__, flush=True)
        except Exception as e:
            print("FAILED", f.__name__, repr(e), flush=True)
