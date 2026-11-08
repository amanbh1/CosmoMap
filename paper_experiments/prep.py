"""Convert raw CAMELS grids to globally standardised log10 float16 arrays.

Statistics (mean/std of log10 field) are computed from TRAINING simulations only,
pooled over all available redshifts, so redshift evolution of the amplitude is
preserved in the standardised data. Run after each new redshift lands; stats are
computed once from the first run's redshifts and then frozen.

    python prep.py            # process every raw redshift not yet prepared
    python prep.py --inspect  # print value ranges of raw files
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from common import ALL_FIELDS, FLOORS, PREP, REDSHIFTS, load_params, prep_path, raw_path, split


def raw_ready(z: float) -> bool:
    expected = 1000 * 128**3 * 4 + 128
    return all(raw_path(f, z).exists() and raw_path(f, z).stat().st_size == expected for f in ALL_FIELDS)


def inspect():
    for z in REDSHIFTS:
        for f in ALL_FIELDS:
            p = raw_path(f, z)
            if not p.exists():
                continue
            a = np.load(p, mmap_mode="r")
            s = np.asarray(a[::50]).ravel()
            print(f"z={z} {f:5s} shape={a.shape} min={s.min():.3e} p0.1={np.percentile(s, 0.1):.3e} "
                  f"median={np.median(s):.3e} max={s.max():.3e} n<=0={np.sum(s <= 0)}")


def compute_stats(zs):
    tr = split()["train"]
    sub = tr[::8]  # 100 training sims are plenty for a mean/std
    stats = {"mean": {}, "std": {}, "redshifts_for_stats": zs}
    for f in ALL_FIELDS:
        vals = []
        for z in zs:
            a = np.load(raw_path(f, z), mmap_mode="r")
            for i in sub:
                vals.append(np.log10(np.maximum(np.asarray(a[i], dtype=np.float64), FLOORS[f]))[::2, ::2, ::2].ravel())
        v = np.concatenate(vals)
        stats["mean"][f], stats["std"][f] = float(v.mean()), float(v.std())
        print(f"{f}: mean={v.mean():.4f} std={v.std():.4f}")
    P = load_params()[tr]
    stats["param_mean"] = P.mean(0).tolist()
    stats["param_std"] = P.std(0).tolist()
    PREP.mkdir(parents=True, exist_ok=True)
    (PREP / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def convert(z, stats):
    for f in ALL_FIELDS:
        out = prep_path(f, z)
        if out.exists():
            continue
        a = np.load(raw_path(f, z), mmap_mode="r")
        tmp = out.with_suffix(".tmp.npy")
        o = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=a.shape)
        m, s = stats["mean"][f], stats["std"][f]
        for i in range(a.shape[0]):
            o[i] = ((np.log10(np.maximum(np.asarray(a[i], dtype=np.float32), FLOORS[f])) - m) / s).astype(np.float16)
        o.flush()
        del o
        tmp.rename(out)
        print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()
    if args.inspect:
        inspect()
        return
    zs = [z for z in REDSHIFTS if raw_ready(z)]
    if not zs:
        print("no complete raw redshifts yet")
        return
    stats_file = PREP / "stats.json"
    stats = json.loads(stats_file.read_text()) if stats_file.exists() else compute_stats(zs)
    for z in zs:
        convert(z, stats)


if __name__ == "__main__":
    main()
