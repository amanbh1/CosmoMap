"""Field normalisation: log10 transform of multiplicative fields followed by
per-patch standardisation (z-score), with exact inversion for recovering
physical units at inference time.

This mirrors the two-stage scheme in the dissertation:

    1.  X_log = log10(max(X, eps))           # for strictly-positive fields
    2.  X'    = (X_log - mu_patch) / sigma_patch

The per-patch statistics are returned alongside the normalised array so that
predictions can be mapped back to physical units.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

#: Strictly-positive ("multiplicative") fields that receive a log10 transform.
#: Signed fields (e.g. velocity) are standardised without the log.
LOG_FIELDS = frozenset({"Mcdm", "Mgas", "T", "ne", "rho", "P", "density"})


class FieldStats(TypedDict):
    mean: float
    std: float
    log: bool


def is_log_field(name: str) -> bool:
    """Whether ``name`` is treated as a multiplicative (log-transformed) field."""
    return name in LOG_FIELDS


def normalize_field(
    x: np.ndarray, name: str, eps: float = 1e-8
) -> tuple[np.ndarray, FieldStats]:
    """Normalise a single field array and return ``(normalised, stats)``."""
    x = np.asarray(x, dtype=np.float32)
    log = is_log_field(name)
    if log:
        x = np.log10(np.maximum(x, eps))
    mean = float(x.mean())
    std = float(x.std()) + eps
    normed = ((x - mean) / std).astype(np.float32)
    return normed, {"mean": mean, "std": std, "log": log}


def denormalize_field(normed: np.ndarray, stats: FieldStats) -> np.ndarray:
    """Invert :func:`normalize_field` using stored per-patch statistics."""
    x = np.asarray(normed, dtype=np.float32) * stats["std"] + stats["mean"]
    if stats["log"]:
        x = np.power(10.0, x)
    return x.astype(np.float32)


def standardize_columns(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score the columns of a 2D array (used for the conditioning vector).

    Constant columns (zero variance) are mapped to zeros rather than NaNs.
    """
    arr = np.asarray(arr, dtype=np.float32)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    return ((arr - mean) / (std + eps)).astype(np.float32)
