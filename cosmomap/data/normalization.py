"""Field normalisation with constants fitted on the training set.

The fields are strictly positive and span several orders of magnitude, so each is
log-transformed (after a small floor that guards against zeros) and then
standardised with a single mean and standard deviation per field::

    x  = log10(max(X, floor))
    x' = (x - mean_X) / std_X

The constants are computed once, from the training samples only, and reused
unchanged for every sample, redshift and split. Amplitude differences between
simulations and epochs are therefore preserved, and predictions are converted
back to physical units with constants that were fixed before any test target was
seen: no statistic of a target field is used at test time.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np

#: Strictly positive fields that receive a log10 transform before standardisation.
LOG_FIELDS = frozenset({"Mcdm", "Vcdm", "Mgas", "T", "ne", "rho", "P", "density"})

#: Floors applied before the logarithm, in the units of the CAMELS Multifield
#: Dataset; they lie well below the values in the data.
DEFAULT_FLOORS: dict[str, float] = {"Mcdm": 1e6, "Vcdm": 1.0, "T": 10.0, "ne": 1e-12}


class FieldStats(TypedDict):
    mean: float
    std: float
    log: bool
    floor: float


#: Normalisation constants of every field, keyed by field name.
Normalization = dict[str, FieldStats]


def is_log_field(name: str) -> bool:
    """Whether ``name`` is log-transformed before standardisation."""
    return name in LOG_FIELDS


def _transform(x: np.ndarray, log: bool, floor: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.log10(np.maximum(x, floor)) if log else x


def fit_normalization(
    fields: Mapping[str, np.ndarray],
    indices: Sequence[int],
    floors: Mapping[str, float] | None = None,
    default_floor: float = 1e-8,
) -> Normalization:
    """Fit one mean and standard deviation per field from the samples ``indices``.

    Pass the *training* indices: the result is then applied to every split.
    ``fields`` maps a field name to an array ``(N, R, R, R)``; ``floors`` gives the
    floor of each log-transformed field (``default_floor`` for fields not listed).
    """
    floors = DEFAULT_FLOORS if floors is None else floors
    indices = [int(i) for i in indices]
    if not indices:
        raise ValueError("cannot fit the normalisation on an empty set of samples")
    stats: Normalization = {}
    for name, arr in fields.items():
        log = is_log_field(name)
        floor = float(floors.get(name, default_floor))
        n, total, total_sq = 0, 0.0, 0.0
        for i in indices:  # one sample at a time: the full training set need not fit in memory twice
            x = _transform(arr[i], log, floor).astype(np.float64)
            n += x.size
            total += float(x.sum())
            total_sq += float(np.square(x).sum())
        mean = total / n
        std = float(np.sqrt(max(total_sq / n - mean**2, 0.0)))
        stats[name] = {"mean": mean, "std": std if std > 0 else 1.0, "log": log, "floor": floor}
    return stats


def normalize_field(x: np.ndarray, stats: FieldStats) -> np.ndarray:
    """Apply fixed normalisation constants to a field in physical units."""
    x = _transform(x, stats["log"], stats["floor"])
    return ((x - stats["mean"]) / stats["std"]).astype(np.float32)


def to_log_physical(normed: np.ndarray, stats: FieldStats) -> np.ndarray:
    """Standardised values -> log10 of the physical field (plain units for non-log fields)."""
    return np.asarray(normed, dtype=np.float32) * stats["std"] + stats["mean"]


def denormalize_field(normed: np.ndarray, stats: FieldStats) -> np.ndarray:
    """Invert :func:`normalize_field`, returning physical units."""
    x = to_log_physical(normed, stats)
    if stats["log"]:
        x = np.power(10.0, x)
    return x.astype(np.float32)


def save_normalization(stats: Normalization, path: str | Path) -> None:
    """Write the constants to JSON, to be reused for evaluation and inference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))


def load_normalization(path: str | Path) -> Normalization:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no normalisation constants at {path}; they are written by training "
            "(normalization.json in the run's output directory)"
        )
    return json.loads(path.read_text())


def standardize_columns(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score the columns of a 2D array (used for the conditioning vector).

    Constant columns (zero variance) are mapped to zeros rather than NaNs.
    """
    arr = np.asarray(arr, dtype=np.float32)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    return ((arr - mean) / (std + eps)).astype(np.float32)
