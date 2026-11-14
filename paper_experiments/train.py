"""Train the 3D FiLM U-Net on globally normalised CAMELS-TNG fields.

    python train.py --name mse_z0_s0 --seed 0 --zs 0.0 --loss mse     # CosmoMap
    python train.py --name film_z0_s0 --seed 0 --zs 0.0               # CosmoMap-sigma (composite likelihood loss)
    python train.py --name nofilm_mse_z0_s0 --seed 0 --zs 0.0 --loss mse --no-cond   # unconditioned ablation

Writes RUNS/<name>/: config.json, history.json (one record per epoch), best.weights.h5,
the weights with the lowest loss on a fixed set of validation sub-volumes (used for every
result in the paper), and last.weights.h5, the weights after the final epoch.
"""

from __future__ import annotations

import argparse
import json
import threading
import queue
import time

import numpy as np
import tensorflow as tf

from common import FIELDS_IN, FIELDS_OUT, RUNS, available_redshifts, cond_vector, load_params, load_stats, prep_path, split
from model import build

LOGVAR_CLIP = 10.0
W = dict(nll=0.408, mse=0.150, mae=0.134, huber=0.207, var=0.01)


def loss_mean_only(y, pred, mse_only=False):
    """Ablation losses on the predicted mean only (no likelihood / variance terms)."""
    n = y.shape[-1]
    e = y - pred[..., :n]
    if mse_only:
        return tf.reduce_mean(e * e)
    ae = tf.abs(e)
    hub = tf.where(ae < 1.0, 0.5 * e * e, ae - 0.5)
    return W["mse"] * tf.reduce_mean(e * e) + W["mae"] * tf.reduce_mean(ae) + W["huber"] * tf.reduce_mean(hub)


def loss_fn(y, pred):
    n = y.shape[-1]
    mu = pred[..., :n]
    lv = tf.clip_by_value(pred[..., n:], -LOGVAR_CLIP, LOGVAR_CLIP)
    e = y - mu
    ae = tf.abs(e)
    hub = tf.where(ae < 1.0, 0.5 * e * e, ae - 0.5)
    nll = 0.5 * tf.reduce_mean(e * e * tf.exp(-lv) + lv)
    return (W["nll"] * nll + W["mse"] * tf.reduce_mean(e * e) + W["mae"] * tf.reduce_mean(ae)
            + W["huber"] * tf.reduce_mean(hub) + W["var"] * tf.reduce_mean(tf.exp(lv)))


def loss_nll(y, pred, beta=0.0):
    """Pure heteroscedastic Gaussian NLL; beta > 0 gives beta-NLL (Seitzer et al. 2022), which weights each
    voxel's NLL by stop_gradient(sigma^(2 beta)) so poorly fitted voxels are not down-weighted."""
    n = y.shape[-1]
    mu = pred[..., :n]
    lv = tf.clip_by_value(pred[..., n:], -LOGVAR_CLIP, LOGVAR_CLIP)
    e = y - mu
    nll = 0.5 * (e * e * tf.exp(-lv) + lv)
    if beta > 0:
        nll = tf.stop_gradient(tf.exp(beta * lv)) * nll
    return tf.reduce_mean(nll)


class Sampler:
    """Random periodic crops with the 48 cube symmetries as augmentation."""

    def __init__(self, sims, zs, patch, stats, use_cond, seed):
        self.sims, self.zs, self.patch = sims, zs, patch
        self.rng = np.random.default_rng(seed)
        self.params = load_params()
        self.stats, self.use_cond = stats, use_cond
        self.arr = {(f, z): np.load(prep_path(f, z), mmap_mode="r") for f in FIELDS_IN + FIELDS_OUT for z in zs}

    def crop(self, sim, z, start, aug=None):
        p = self.patch
        idx = [np.arange(s, s + p) % 128 for s in start]
        chans = []
        for f in FIELDS_IN + FIELDS_OUT:
            a = self.arr[(f, z)][sim]
            # contiguous read when no wrap, else fancy-index
            if all(s + p <= 128 for s in start):
                c = np.asarray(a[start[0]:start[0] + p, start[1]:start[1] + p, start[2]:start[2] + p])
            else:
                c = np.asarray(a[np.ix_(*idx)])
            chans.append(c.astype(np.float32))
        x = np.stack(chans, -1)
        if aug is not None:
            perm, flips = aug
            x = np.transpose(x, (*perm, 3))
            for ax in range(3):
                if flips[ax]:
                    x = x[::-1] if ax == 0 else (x[:, ::-1] if ax == 1 else x[:, :, ::-1])
            x = np.ascontiguousarray(x)
        c = cond_vector(self.params[sim], z, self.stats, self.use_cond)
        return x[..., :2], x[..., 2:], c

    def random(self):
        sim = int(self.rng.choice(self.sims))
        z = float(self.zs[self.rng.integers(len(self.zs))])
        start = self.rng.integers(0, 128, 3)
        aug = (tuple(self.rng.permutation(3)), self.rng.integers(0, 2, 3).astype(bool))
        return self.crop(sim, z, start, aug)


def batches(sampler, bs, n_workers=6, depth=24):
    q = queue.Queue(depth)

    def work():
        while True:
            q.put(sampler.random())

    for _ in range(n_workers):
        threading.Thread(target=work, daemon=True).start()
    while True:
        items = [q.get() for _ in range(bs)]
        yield (np.stack([i[0] for i in items]), np.stack([i[2] for i in items])), np.stack([i[1] for i in items])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cond", action="store_true")
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--zs", type=str, default="")
    ap.add_argument("--loss", default="composite", choices=["composite", "nounc", "mse", "nll", "betanll"])
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--norm", default="group", choices=["group", "batch"])
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    stats = load_stats()
    zs = [float(z) for z in args.zs.split(",")] if args.zs else available_redshifts()
    sp = split()
    use_cond = not args.no_cond
    print("redshifts", zs, "cond", use_cond, flush=True)
    (out / "config.json").write_text(json.dumps(vars(args) | {"zs": zs}, indent=1))  # also needed if training is interrupted

    tr = Sampler(sp["train"], zs, args.patch, stats, use_cond, args.seed)
    va = Sampler(sp["val"], zs, args.patch, stats, use_cond, 12345)
    val = [va.random() for _ in range(320)]  # fixed validation crops
    vx = np.stack([v[0] for v in val]); vc = np.stack([v[2] for v in val]); vy = np.stack([v[1] for v in val])

    model = build(base=args.base, groups=args.groups, norm=args.norm)
    print("params", model.count_params(), flush=True)
    LOSS = {"composite": loss_fn, "nounc": loss_mean_only, "mse": lambda y, p: loss_mean_only(y, p, mse_only=True),
            "nll": loss_nll, "betanll": lambda y, p: loss_nll(y, p, beta=0.5)}[args.loss]
    sched = tf.keras.optimizers.schedules.CosineDecay(args.lr / 10, args.steps * (args.epochs - 1), alpha=0.02,
                                                      warmup_target=args.lr, warmup_steps=args.steps)
    opt = tf.keras.optimizers.AdamW(sched, weight_decay=1e-5, clipnorm=1.0)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    @tf.function
    def step(x, c, y):
        with tf.GradientTape() as tape:
            pred = model({"image": x, "cosmo": c}, training=True)
            loss = LOSS(y, tf.cast(pred, tf.float32))
            scaled = opt.scale_loss(loss)
        g = tape.gradient(scaled, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    @tf.function
    def evalb(x, c, y):
        pred = tf.cast(model({"image": x, "cosmo": c}, training=False), tf.float32)
        n = y.shape[-1]
        return LOSS(y, pred), tf.reduce_mean(tf.square(y - pred[..., :n]), axis=[0, 1, 2, 3])

    def validate():
        ls, ms = [], []
        for i in range(0, len(vx), 8):
            l, m = evalb(vx[i:i + 8], vc[i:i + 8], vy[i:i + 8])
            ls.append(float(l)); ms.append(m.numpy())
        return float(np.mean(ls)), np.mean(ms, 0)

    gen = batches(tr, args.bs)
    best, bad, hist = np.inf, 0, []
    for ep in range(args.epochs):
        t0 = time.time(); tl = []
        for _ in range(args.steps):
            (x, c), y = next(gen)
            tl.append(float(step(x, c, y)))
        vl, vmse = validate()
        rec = dict(epoch=ep + 1, train=float(np.mean(tl)), val=vl, val_mse_T=float(vmse[0]),
                   val_mse_ne=float(vmse[1]), sec=time.time() - t0)
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        (out / "history.json").write_text(json.dumps(hist, indent=1))
        if vl < best - 1e-4:
            best, bad = vl, 0
            model.save_weights(out / "best.weights.h5")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop", flush=True)
                break
    model.save_weights(out / "last.weights.h5")  # final-epoch weights, to compare with the selected checkpoint
    (out / "config.json").write_text(json.dumps(vars(args) | {"zs": zs}, indent=1))
    print("done best", best, flush=True)


if __name__ == "__main__":
    main()
