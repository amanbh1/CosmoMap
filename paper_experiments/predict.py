"""Predict full 128^3 test boxes with a trained U-Net (periodic tiling).

Default (used for all results in the paper): each box is tiled with periodic 64^3
windows, the size of the training sub-volumes, keeping the central 32^3 of each
(16-voxel margin), so group-normalisation statistics match training.
--core/--margin change the tiling; --periodic predicts the full box in one pass.

Outputs (float16, log10 physical units) in RUNS/<name>/pred/:
    mu_<field>_z<z>.npy     (n_test, 128,128,128)
    sig_<field>_z<z>.npy    total sigma in dex (aleatoric + MC-dropout epistemic)
    sige_<field>_z<z>.npy   epistemic sigma in dex
The sigma files are meaningful only for models trained with a likelihood term (CosmoMap-sigma and
the NLL variants). For the others (--loss mse, nounc) the variance channels of the network are
untrained, and nothing in the paper uses their sigma files.

    python predict.py --name film_z0_s0 [--mc 8]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import tensorflow as tf

from common import FIELDS_IN, FIELDS_OUT, FLOORS, RUNS, cond_vector, load_params, load_stats, prep_path, split
from model import build

M, C = 16, 32  # margin, core (paper default: 64^3 windows)


def windows(box):
    """box: (128,128,128,C) -> (8, C+2M, C+2M, C+2M, C) periodic windows and their core starts."""
    ws, starts = [], []
    for sx in range(0, 128, C):
        for sy in range(0, 128, C):
            for sz in range(0, 128, C):
                ix = [np.arange(s - M, s + C + M) % 128 for s in (sx, sy, sz)]
                ws.append(box[np.ix_(*ix)])
                starts.append((sx, sy, sz))
    return np.stack(ws), starts


def assemble(outs, starts, nch):
    full = np.zeros((128, 128, 128, nch), np.float32)
    for o, (sx, sy, sz) in zip(outs, starts):
        full[sx:sx + C, sy:sy + C, sz:sz + C] = o[M:M + C, M:M + C, M:M + C]
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mc", type=int, default=8, help="MC-dropout passes (0 = off)")
    ap.add_argument("--zs", type=str, default="")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--margin", type=int, default=16, help="periodic margin around each core")
    ap.add_argument("--core", type=int, default=32, help="core size of each tile (divides 128)")
    ap.add_argument("--as-run", default="", help="write predictions as a new run directory (for tests)")
    ap.add_argument("--nbody", default="", help="raw N-body Mtot grid to use as the density input")
    ap.add_argument("--nbody-vel", default="", help="N-body velocity-modulus grid (rows in test-split order, from grid_velocity.py) to use as the velocity input")
    ap.add_argument("--periodic", action="store_true", help="exact full-box inference with periodic padding (no tiling)")
    args = ap.parse_args()
    global M, C
    C = args.core
    M = args.margin
    bs = 8 if C + 2 * M <= 64 else (2 if C + 2 * M <= 96 else 1)
    ntile = (128 // C) ** 3
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    run = RUNS / args.name
    cfg = json.loads((run / "config.json").read_text())
    zs = [float(z) for z in args.zs.split(",")] if args.zs else cfg["zs"]
    use_cond = not cfg["no_cond"]
    stats = load_stats()
    params = load_params()
    test = split()[args.split]
    if args.periodic:
        from model_periodic import load_periodic
        model = load_periodic(run / "best.weights.h5", base=cfg.get("base", 64), groups=cfg.get("groups", 8), norm=cfg.get("norm", "group"))
    else:
        model = build(base=cfg.get("base", 64), groups=cfg.get("groups", 8), norm=cfg.get("norm", "group"))
        model.load_weights(run / "best.weights.h5")
    infer = tf.function(lambda x, c, t: model({"image": x, "cosmo": c}, training=t), reduce_retracing=True)
    if args.as_run:
        newrun = RUNS / args.as_run
        newrun.mkdir(exist_ok=True)
        (newrun / "config.json").write_text(json.dumps(cfg | {"zs": zs, "source_run": args.name, "margin": M, "core": C,
                                                             "periodic": args.periodic, "nbody": args.nbody, "nbody_vel": args.nbody_vel}))
        outdir = newrun / "pred"
    else:
        outdir = run / ("pred" if args.split == "test" else "pred_val")
    outdir.mkdir(exist_ok=True)
    nb = np.load(args.nbody, mmap_mode="r") if args.nbody else None
    nbv = np.load(args.nbody_vel, mmap_mode="r") if args.nbody_vel else None
    OMEGA_B = 0.049  # CAMELS baryon density: scale N-body total matter to the dark matter share
    n = len(FIELDS_OUT)

    for z in zs:
        arr = {f: np.load(prep_path(f, z), mmap_mode="r") for f in FIELDS_IN}
        mu_o = {f: np.lib.format.open_memmap(outdir / f"mu_{f}_z{z:.1f}.npy", "w+", np.float16, (len(test), 128, 128, 128)) for f in FIELDS_OUT}
        sg_o = {f: np.lib.format.open_memmap(outdir / f"sig_{f}_z{z:.1f}.npy", "w+", np.float16, (len(test), 128, 128, 128)) for f in FIELDS_OUT}
        se_o = {f: np.lib.format.open_memmap(outdir / f"sige_{f}_z{z:.1f}.npy", "w+", np.float16, (len(test), 128, 128, 128)) for f in FIELDS_OUT}
        for k, sim in enumerate(test):
            box = np.stack([np.asarray(arr[f][sim], np.float32) for f in FIELDS_IN], -1)
            if nb is not None:
                om = float(params[sim, 0])
                # N-body grids store mass per voxel; hydro grids store density -> divide by voxel volume
                rho = np.asarray(nb[sim], np.float64) / (25.0 / 128) ** 3 * (om - OMEGA_B) / om
                box[..., 0] = ((np.log10(np.maximum(rho, 1e6)) - stats["mean"]["Mcdm"]) / stats["std"]["Mcdm"]).astype(np.float32)
            if nbv is not None:
                v = np.asarray(nbv[k], np.float64)
                box[..., 1] = ((np.log10(np.maximum(v, FLOORS["Vcdm"])) - stats["mean"]["Vcdm"]) / stats["std"]["Vcdm"]).astype(np.float32)
            if args.periodic:
                full = box[None]
                c = cond_vector(params[sim], z, stats, use_cond)[None]

                def run_pass(training):
                    return infer(full, c, training).numpy().astype(np.float32)[0]
            else:
                w, starts = windows(box)
                c = np.repeat(cond_vector(params[sim], z, stats, use_cond)[None], bs, 0)

                def run_pass(training):
                    outs = []
                    for i in range(0, ntile, bs):
                        outs.append(infer(w[i:i + bs], c, training).numpy().astype(np.float32))
                    return assemble(np.concatenate(outs), starts, 2 * n)

            det = run_pass(False)
            mu = det[..., :n]
            var_a = np.exp(np.clip(det[..., n:], -10, 10))
            if args.mc > 0:
                mcs = np.stack([run_pass(True)[..., :n] for _ in range(args.mc)])
                var_e = mcs.var(0)
            else:
                var_e = np.zeros_like(var_a)
            for j, f in enumerate(FIELDS_OUT):
                s = stats["std"][f]
                mu_o[f][k] = (mu[..., j] * s + stats["mean"][f]).astype(np.float16)
                sg_o[f][k] = (np.sqrt(var_a[..., j] + var_e[..., j]) * s).astype(np.float16)
                se_o[f][k] = (np.sqrt(var_e[..., j]) * s).astype(np.float16)
            if k % 10 == 0:
                print(f"z={z} {k}/{len(test)}", flush=True)
        for d in (mu_o, sg_o, se_o):
            for v in d.values():
                v.flush()
    print("done", flush=True)


if __name__ == "__main__":
    main()
