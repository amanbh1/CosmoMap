"""Score saved predictions against the truth on the fixed 100-simulation test set.

All metrics are in absolute log10 physical units (dex) unless stated.

    python metrics.py film_s0 nofilm_s0 binned gbt
Writes RUNS/<name>/metrics.json (per-sim records + summary).
"""

from __future__ import annotations

import json
import sys

import numpy as np

from common import BOX, FIELDS_OUT, RUNS, load_params, load_stats, prep_path, split, to_log_physical

REGIMES = [("deep_voids", 0, 10), ("voids", 10, 30), ("sheets", 30, 50), ("filaments", 50, 80),
           ("outskirts", 80, 95), ("cores", 95, 100)]
NK = 20


def kgrid():
    k1 = np.fft.fftfreq(128, d=BOX / 128) * 2 * np.pi
    kz = np.fft.rfftfreq(128, d=BOX / 128) * 2 * np.pi
    kk = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2)
    edges = np.logspace(np.log10(2 * np.pi / BOX), np.log10(np.pi * 128 / BOX), NK + 1)
    idx = np.digitize(kk.ravel(), edges) - 1
    return idx, edges


KIDX, KEDGES = kgrid()


def pk_cross(a, b):
    """Auto/cross power of overdensity fields a, b (linear units) binned in k."""
    da, db = a / a.mean() - 1, b / b.mean() - 1
    fa, fb = np.fft.rfftn(da), np.fft.rfftn(db)
    out = []
    for x in (fa * fa.conj(), fb * fb.conj(), fa * fb.conj()):
        v = np.real(x).ravel()
        ok = (KIDX >= 0) & (KIDX < NK)
        s = np.bincount(KIDX[ok], weights=v[ok], minlength=NK)
        c = np.bincount(KIDX[ok], minlength=NK)
        out.append(s / np.maximum(c, 1))
    return out  # P_true, P_pred, P_cross


def r2(t, p):
    return float(1 - np.mean((t - p) ** 2) / np.var(t))


def score_one(t, p, sig, rho_pct):
    """t, p: dict field -> log10 arrays (128^3); sig: dict or None."""
    rec = {}
    for f in FIELDS_OUT:
        e = p[f] - t[f]
        r = dict(r2=r2(t[f], p[f]), rmse=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean()),
                 pearson=float(np.corrcoef(t[f].ravel(), p[f].ravel())[0, 1]))
        for name, lo, hi in REGIMES:
            m = (rho_pct >= lo) & (rho_pct < hi) if hi < 100 else rho_pct >= lo
            r[f"{name}_r2"] = r2(t[f][m], p[f][m])
            r[f"{name}_rmse"] = float(np.sqrt(np.mean(e[m] ** 2)))
            r[f"{name}_bias"] = float(e[m].mean())
        if sig is not None:
            s = np.maximum(sig[f], 1e-4)
            r["cov68"] = float(np.mean(np.abs(e) < s))
            r["cov95"] = float(np.mean(np.abs(e) < 1.96 * s))
            r["mean_sigma"] = float(s.mean())
        Pt, Pp, Px = pk_cross(10.0 ** t[f].astype(np.float64), 10.0 ** p[f].astype(np.float64))
        r["Tk"] = np.sqrt(Pp / Pt).tolist()
        r["rk"] = (Px / np.sqrt(Pt * Pp)).tolist()
        rec[f] = r
    # bremsstrahlung emissivity  eps ∝ ne^2 T^1/2  (log10, constant dropped)
    let = 2 * t["ne"] + 0.5 * t["T"]
    lep = 2 * p["ne"] + 0.5 * p["T"]
    et, ep = 10.0 ** let.astype(np.float64), 10.0 ** lep.astype(np.float64)
    proj_t, proj_p = np.log10(et.sum(0)), np.log10(ep.sum(0))
    top = let >= np.percentile(let, 99)
    Pt, Pp, Px = pk_cross(et, ep)
    rec["eps"] = dict(
        r2_log=r2(let, lep),
        L_ratio=float(ep.sum() / et.sum()),
        proj_pearson_log=float(np.corrcoef(proj_t.ravel(), proj_p.ravel())[0, 1]),
        Tew_true=float((et * 10.0 ** t["T"]).sum() / et.sum()),
        Tew_pred=float((ep * 10.0 ** p["T"]).sum() / ep.sum()),
        top1_T_r2=r2(t["T"][top], p["T"][top]), top1_ne_r2=r2(t["ne"][top], p["ne"][top]),
        top1_T_bias=float((p["T"] - t["T"])[top].mean()), top1_ne_bias=float((p["ne"] - t["ne"])[top].mean()),
        Tk=np.sqrt(Pp / Pt).tolist(), rk=(Px / np.sqrt(Pt * Pp)).tolist(),
    )
    return rec


def summarise(records):
    out = {}
    for z in sorted({r["z"] for r in records}):
        rs = [r for r in records if r["z"] == z]
        s = {}
        for grp in FIELDS_OUT + ["eps"]:
            for key in rs[0][grp]:
                vals = np.array([r[grp][key] for r in rs], dtype=float)
                s[f"{grp}.{key}"] = [np.mean(vals, 0).tolist(), np.std(vals, 0).tolist()]
        out[f"{z:.1f}"] = s
    return out


def main():
    stats = load_stats()
    test = split()["test"]
    params = load_params()
    for name in sys.argv[1:]:
        run = RUNS / name
        zs = json.loads((run / "config.json").read_text())["zs"]
        records = []
        for z in zs:
            truth = {f: np.load(prep_path(f, z), mmap_mode="r") for f in FIELDS_OUT + ["Mcdm"]}
            mu = {f: np.load(run / "pred" / f"mu_{f}_z{z:.1f}.npy", mmap_mode="r") for f in FIELDS_OUT}
            sigp = run / "pred" / f"sig_{FIELDS_OUT[0]}_z{z:.1f}.npy"
            sg = {f: np.load(run / "pred" / f"sig_{f}_z{z:.1f}.npy", mmap_mode="r") for f in FIELDS_OUT} if sigp.exists() else None
            for k, sim in enumerate(test):
                t = {f: to_log_physical(np.asarray(truth[f][sim], np.float32), f, stats) for f in FIELDS_OUT}
                p = {f: np.asarray(mu[f][k], np.float32) for f in FIELDS_OUT}
                s = {f: np.asarray(sg[f][k], np.float32) for f in FIELDS_OUT} if sg is not None else None
                rho = np.asarray(truth["Mcdm"][sim], np.float32)
                pct = np.argsort(np.argsort(rho.ravel())).reshape(rho.shape) / rho.size * 100
                rec = score_one(t, p, s, pct)
                rec.update(sim=int(sim), z=z, params=params[sim].tolist())
                records.append(rec)
            print(name, "z", z, "done", flush=True)
        res = dict(k_centres=np.sqrt(KEDGES[1:] * KEDGES[:-1]).tolist(), summary=summarise(records), records=records)
        (run / "metrics.json").write_text(json.dumps(res))
        for z, s in res["summary"].items():
            print(name, z, {k: round(v[0], 3) for k, v in s.items() if not isinstance(v[0], list) and
                            (k.endswith(".r2") or k.endswith(".rmse") or "cov95" in k or "L_ratio" in k or k.endswith("cores_r2"))})


if __name__ == "__main__":
    main()
