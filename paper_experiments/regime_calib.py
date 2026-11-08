"""Coverage of CosmoMap-sigma intervals by environment (first seed, z=0, 100 test boxes): all voxels, densest
5 per cent by dark matter density, brightest 1 per cent by true emissivity. Raw sigma and sigma rescaled by the
validation factor in calibration.json; also the pairing with CosmoMap (MSE) means. Writes RUNS/regime_calibration.json."""
import json

import numpy as np
from scipy.special import erfinv
from scipy.stats import spearmanr

from common import RUNS, load_stats, prep_path, split, to_log_physical

z = 0.0
st = load_stats()
levels = np.array([0.5, 0.68, 0.9, 0.95])
zq = np.sqrt(2) * erfinv(levels)
sims = split()["test"][:100]
cal = json.load(open(RUNS / "film_z0_s0" / "calibration.json"))
tr = {f: np.load(prep_path(f, z), mmap_mode="r") for f in ("T", "ne", "Mcdm")}
mu = {(r, f): np.load(RUNS / r / "pred" / f"mu_{f}_z{z:.1f}.npy", mmap_mode="r") for r in ("film_z0_s0", "mse_z0_s0") for f in ("T", "ne")}
sg = {f: np.load(RUNS / "film_z0_s0" / "pred" / f"sig_{f}_z{z:.1f}.npy", mmap_mode="r") for f in ("T", "ne")}
acc = {}
for k, s in enumerate(sims):
    t = {f: to_log_physical(np.asarray(tr[f][s], np.float32), f, st) for f in ("T", "ne")}
    rho = np.asarray(tr["Mcdm"][s], np.float32)
    let = 2 * t["ne"] + 0.5 * t["T"]
    masks = {"all": np.ones(rho.shape, bool), "dense5": rho >= np.percentile(rho, 95), "bright1": let >= np.percentile(let, 99)}
    for f in ("T", "ne"):
        sig = np.maximum(np.asarray(sg[f][k], np.float32), 1e-4)
        for r in ("film_z0_s0", "mse_z0_s0"):
            res = np.abs(np.asarray(mu[(r, f)][k], np.float32) - t[f])
            for m, msk in masks.items():
                sub = slice(None, None, 8) if m == "all" else slice(None)
                acc.setdefault((r, f, m), [[], []])
                acc[(r, f, m)][0].append(res[msk][sub]); acc[(r, f, m)][1].append(sig[msk][sub])
out = {}
for (r, f, m), (rr, ss) in acc.items():
    rr, ss = np.concatenate(rr), np.concatenate(ss)
    u, c = rr / ss, cal[f]["scale"]
    d = dict(raw=[float((u < q).mean()) for q in zq], cal=[float((u < c * q).mean()) for q in zq],
             spearman=float(spearmanr(rr[::20], ss[::20])[0]), med_sigma=float(np.median(ss)), rms_res=float(np.sqrt(np.mean(rr ** 2))))
    out[f"{r}|{f}|{m}"] = d
    print(f"{r:11s} {f:2s} {m:7s} raw " + " ".join(f"{x:.3f}" for x in d["raw"]) + " | cal " + " ".join(f"{x:.3f}" for x in d["cal"])
          + f" | spearman {d['spearman']:.2f} med_sig {d['med_sigma']:.3f} rms_res {d['rms_res']:.3f}")
out["levels"] = levels.tolist()
(RUNS / "regime_calibration.json").write_text(json.dumps(out, indent=1))
