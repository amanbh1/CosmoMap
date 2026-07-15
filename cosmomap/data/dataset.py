"""Patch-based ``tf.data`` pipeline.

During training a random cubic patch is cropped from each sub-box (data
augmentation through position); during evaluation a centred patch is used for
reproducibility. Each field is normalised per patch (see
:mod:`cosmomap.data.normalization`).

The training dataset is *infinite* (it repeats), so callers pass
``steps_per_epoch``; the evaluation dataset is finite (one pass).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import tensorflow as tf

from cosmomap.config import Config
from cosmomap.data.container import CamelsData
from cosmomap.data.normalization import normalize_field


def split_indices(
    n: int, split: Sequence[float], seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle ``range(n)`` and split into (train, val, test) index arrays."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    train_frac, val_frac, _ = split
    a = int(n * train_frac)
    b = a + int(n * val_frac)
    return idx[:a], idx[a:b], idx[b:]


def split_simulations(
    sim_ids: np.ndarray, split: Sequence[float], seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulation-level split: assign whole simulations (all their redshifts) to
    one of train/val/test, returning the corresponding *sample* indices.

    This reproduces the thesis's no-leakage scheme (§3.2.5, 800/100/100 sims):
    patches from the same simulation never appear in more than one split.
    """
    sim_ids = np.asarray(sim_ids)
    unique = np.unique(sim_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique)
    train_frac, val_frac, _ = split
    a = int(len(perm) * train_frac)
    b = a + int(len(perm) * val_frac)
    train_sims, val_sims, test_sims = perm[:a], perm[a:b], perm[b:]
    all_idx = np.arange(len(sim_ids))
    return (
        all_idx[np.isin(sim_ids, train_sims)],
        all_idx[np.isin(sim_ids, val_sims)],
        all_idx[np.isin(sim_ids, test_sims)],
    )


def _patch_start(r: int, p: int, rng: np.random.Generator, training: bool) -> tuple[int, int, int]:
    if r <= p:
        return (0, 0, 0)
    if training:
        return tuple(int(rng.integers(0, r - p + 1)) for _ in range(3))  # type: ignore[return-value]
    s = (r - p) // 2
    return (s, s, s)


def _generator(
    data: CamelsData, indices: Sequence[int], config: Config, training: bool, seed: int
) -> Iterator[tuple[dict[str, np.ndarray], np.ndarray]]:
    rng = np.random.default_rng(seed)
    p = config.data.patch_size
    r = data.resolution
    fin, fout = config.data.fields_in, config.data.fields_out
    eps = config.data.log_epsilon
    order = np.asarray(indices)

    while True:
        for idx in rng.permutation(order) if training else order:
            i, j, k = _patch_start(r, p, rng, training)
            sl = (slice(i, i + p), slice(j, j + p), slice(k, k + p))
            x = np.stack(
                [normalize_field(data.fields[f][idx][sl], f, eps)[0] for f in fin], axis=-1
            )
            y = np.stack(
                [normalize_field(data.fields[f][idx][sl], f, eps)[0] for f in fout], axis=-1
            )
            yield (
                {"image": x.astype(np.float32), "cosmo": np.asarray(data.cond[idx], np.float32)},
                y.astype(np.float32),
            )
        if not training:
            break


def make_dataset(
    data: CamelsData, indices: Sequence[int], config: Config, training: bool
) -> tf.data.Dataset:
    """Build a batched, prefetched ``tf.data.Dataset`` over the given samples."""
    p = config.data.patch_size
    signature = (
        {
            "image": tf.TensorSpec(shape=(p, p, p, config.n_in), dtype=tf.float32),
            "cosmo": tf.TensorSpec(shape=(config.n_cond,), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(p, p, p, config.n_out), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda: _generator(data, list(indices), config, training, config.train.seed),
        output_signature=signature,
    )
    if training:
        # The generator already loops forever; marking the dataset INFINITE lets
        # Keras pair it with steps_per_epoch without a "ran out of data" warning.
        ds = ds.repeat()
    return ds.batch(config.train.batch_size).prefetch(tf.data.AUTOTUNE)
