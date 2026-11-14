"""Measure wall-clock time to predict one full 128^3 box with the scheme used in the paper:
64 periodic 64^3 windows (central 32^3 of each kept), in batches of 8, as in predict.py.
The time includes cutting the windows and assembling the box, not reading the inputs from disk."""
import time

import numpy as np
import tensorflow as tf

from common import MAIN, RUNS, cond_vector, load_params, load_stats, prep_path, split
from model import build
from predict import assemble, windows

BS = 8  # batch size predict.py uses for 64^3 windows
tf.keras.mixed_precision.set_global_policy("mixed_float16")
st = load_stats(); P = load_params(); test = split()["test"]
m = build(); m.load_weights(RUNS / MAIN / "best.weights.h5")
infer = tf.function(lambda x, c: m({"image": x, "cosmo": c}, training=False), reduce_retracing=True)
arr = {f: np.load(prep_path(f, 0.0), mmap_mode="r") for f in ("Mcdm", "Vcdm")}
times, gpu = [], []
for k in range(12):
    sim = test[k]
    box = np.stack([np.asarray(arr[f][sim], np.float32) for f in ("Mcdm", "Vcdm")], -1)
    t0 = time.perf_counter()
    w, starts = windows(box)
    assert len(w) % BS == 0
    c = np.repeat(cond_vector(P[sim], 0.0, st)[None], BS, 0)
    t1 = time.perf_counter()
    outs = [infer(w[i:i + BS], c).numpy() for i in range(0, len(w), BS)]
    t2 = time.perf_counter()
    assemble(np.concatenate(outs).astype(np.float32), starts, 4)
    times.append(time.perf_counter() - t0); gpu.append(t2 - t1)
print(f"{len(w)} windows of {w.shape[1]}^3 per box")
print("per-box seconds (excluding first 2 warm-up):", np.round(times[2:], 2), "median", round(float(np.median(times[2:])), 2))
print("of which network forward passes: median", round(float(np.median(gpu[2:])), 2))
