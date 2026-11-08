"""Bright-structure detection in 3D (z=0 test set): completeness AND purity, with the lenient any-overlap
criterion of validation_extra.py and a strict one (the matching structure must cover at least half of the
other's voxels). Same absolute threshold as validation_extra.json.   Writes RUNS/structure_purity.json

    python structure_purity.py
"""
import json

import numpy as np
from scipy import ndimage

from common import GBT, MAIN, NOFILM, RUNS, load_stats, prep_path, split, to_log_physical

Z = 0.0
MODELS = {"film": MAIN, "nofilm": NOFILM, "gbt": GBT}
STRUCT = np.ones((3, 3, 3), bool)


def match(lab_a, n_a, mask_b):
    """For each component of lab_a: (any overlap with mask_b, fraction of its voxels inside mask_b)."""
    if n_a == 0:
        return np.zeros(0, bool), np.zeros(0)
    idx = np.arange(1, n_a + 1)
    frac = ndimage.mean(mask_b, lab_a, index=idx)
    return frac > 0, frac


def main():
    st = load_stats()
    test = split()["test"]
    thr = json.loads((RUNS / "validation_extra.json").read_text())["threshold_log_eps"]
    tr = {f: np.load(prep_path(f, Z), mmap_mode="r") for f in ("T", "ne")}
    pr = {m: {f: np.load(RUNS / r / "pred" / f"mu_{f}_z{Z:.1f}.npy", mmap_mode="r") for f in ("T", "ne")} for m, r in MODELS.items()}
    acc = {m: dict(n_true=0, n_pred=0, comp_any=0, comp_half=0, pur_any=0, pur_half=0) for m in MODELS}
    for k, sim in enumerate(test):
        lt = 2 * to_log_physical(np.asarray(tr["ne"][sim], np.float32), "ne", st) + 0.5 * to_log_physical(np.asarray(tr["T"][sim], np.float32), "T", st)
        mt = lt > thr
        lab_t, n_t = ndimage.label(mt, structure=STRUCT)
        for m in MODELS:
            lp = 2 * np.asarray(pr[m]["ne"][k], np.float32) + 0.5 * np.asarray(pr[m]["T"][k], np.float32)
            mp = lp > thr
            lab_p, n_p = ndimage.label(mp, structure=STRUCT)
            ht, ft = match(lab_t, n_t, mp)   # true structures found by the prediction
            hp, fp = match(lab_p, n_p, mt)   # predicted structures that are real
            a = acc[m]
            a["n_true"] += n_t; a["n_pred"] += n_p
            a["comp_any"] += int(ht.sum()); a["comp_half"] += int((ft >= 0.5).sum())
            a["pur_any"] += int(hp.sum()); a["pur_half"] += int((fp >= 0.5).sum())
        if k % 20 == 0:
            print("box", k, flush=True)
    out = {}
    for m, a in acc.items():
        out[m] = a | dict(completeness_any=a["comp_any"] / a["n_true"], completeness_half=a["comp_half"] / a["n_true"],
                          purity_any=a["pur_any"] / max(a["n_pred"], 1), purity_half=a["pur_half"] / max(a["n_pred"], 1))
        print(m, {k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in out[m].items()})
    (RUNS / "structure_purity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
