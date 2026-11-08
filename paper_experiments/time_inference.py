"""Measure wall-clock time to predict one full 128^3 box (8 periodic 96^3 windows), as in predict.py."""
import time

import numpy as np
import tensorflow as tf

from common import MAIN, RUNS, cond_vector, load_params, load_stats, prep_path, split
from model import build
from predict import assemble, windows

tf.keras.mixed_precision.set_global_policy("mixed_float16")
st = load_stats(); P = load_params(); test = split()["test"]
m = build(); m.load_weights(RUNS / MAIN / "best.weights.h5")
infer = tf.function(lambda x, c: m({"image": x, "cosmo": c}, training=False), reduce_retracing=True)
arr = {f: np.load(prep_path(f, 0.0), mmap_mode="r") for f in ("Mcdm", "Vcdm")}
times = []
for k in range(12):
    sim = test[k]
    box = np.stack([np.asarray(arr[f][sim], np.float32) for f in ("Mcdm", "Vcdm")], -1)
    t0 = time.perf_counter()
    w, starts = windows(box)
    c = np.repeat(cond_vector(P[sim], 0.0, st)[None], 2, 0)
    outs = [infer(w[i:i + 2], c).numpy() for i in range(0, 8, 2)]
    assemble(np.concatenate(outs).astype(np.float32), starts, 4)
    times.append(time.perf_counter() - t0)
print("per-box seconds (excluding first 2 warm-up):", np.round(times[2:], 3), "median", round(float(np.median(times[2:])), 3))
