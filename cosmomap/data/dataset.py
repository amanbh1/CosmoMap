"""Patch-based ``tf.data`` pipeline.

During training a random cubic patch is cut from each box with periodic wrapping
and one of the 48 symmetries of the cube is applied; these are exact symmetries
of the data because the velocity input is a magnitude. During evaluation a
centred patch is used for reproducibility. Every field is normalised with the
fixed training-set constants of :mod:`cosmomap.data.normalization`.

The training dataset is *infinite* (it repeats), so callers pass
``steps_per_epoch``; the evaluation dataset is finite (one pass).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import tensorflow as tf

from cosmomap.config import Config
from cosmomap.data.container import CamelsData
from cosmomap.data.normalization import Normalization, normalize_field


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

    Snapshots of one simulation are correlated, so keeping them in a single split
    prevents information leaking between splits.
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


def crop_periodic(box: np.ndarray, start: Sequence[int], size: int) -> np.ndarray:
    """Cut a ``size``-cubed patch starting at ``start`` from a periodic box."""
    idx = [np.arange(s, s + size) % box.shape[a] for a, s in enumerate(start)]
    return box[np.ix_(*idx)]


def cube_symmetry(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply one of the 48 symmetries of the cube (axis permutation and reflections)
    to the three leading axes of ``x``."""
    x = np.transpose(x, (*rng.permutation(3), *range(3, x.ndim)))
    for axis in range(3):
        if rng.integers(2):
            x = np.flip(x, axis)
    return np.ascontiguousarray(x)


def _generator(
    data: CamelsData,
    indices: Sequence[int],
    config: Config,
    training: bool,
    seed: int,
    stats: Normalization,
) -> Iterator[tuple[dict[str, np.ndarray], np.ndarray]]:
    rng = np.random.default_rng(seed)
    p = config.data.patch_size
    r = data.resolution
    fin, fout = config.data.fields_in, config.data.fields_out
    order = np.asarray(indices)
    n_in = len(fin)

    while True:
        for idx in rng.permutation(order) if training else order:
            start = rng.integers(0, r, 3) if training else [max((r - p) // 2, 0)] * 3
            patch = np.stack(
                [
                    normalize_field(crop_periodic(data.fields[f][idx], start, p), stats[f])
                    for f in fin + fout
                ],
                axis=-1,
            )
            if training:
                patch = cube_symmetry(patch, rng)
            yield (
                {"image": patch[..., :n_in], "cosmo": np.asarray(data.cond[idx], np.float32)},
                patch[..., n_in:],
            )
        if not training:
            break


def make_dataset(
    data: CamelsData,
    indices: Sequence[int],
    config: Config,
    training: bool,
    stats: Normalization,
) -> tf.data.Dataset:
    """Build a batched, prefetched ``tf.data.Dataset`` over the given samples.

    ``stats`` are the normalisation constants from
    :func:`cosmomap.data.normalization.fit_normalization`, fitted on the training
    samples; the same constants must be used for every split.
    """
    missing = [f for f in config.data.fields_in + config.data.fields_out if f not in stats]
    if missing:
        raise KeyError(f"normalisation constants are missing for fields {missing}")
    p = config.data.patch_size
    if p > data.resolution:
        raise ValueError(f"patch_size ({p}) exceeds the box resolution ({data.resolution})")
    signature = (
        {
            "image": tf.TensorSpec(shape=(p, p, p, config.n_in), dtype=tf.float32),
            "cosmo": tf.TensorSpec(shape=(config.n_cond,), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(p, p, p, config.n_out), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda: _generator(data, list(indices), config, training, config.train.seed, stats),
        output_signature=signature,
    )
    if training:
        # The generator already loops forever; marking the dataset INFINITE lets
        # Keras pair it with steps_per_epoch without a "ran out of data" warning.
        ds = ds.repeat()
    return ds.batch(config.train.batch_size).prefetch(tf.data.AUTOTUNE)
