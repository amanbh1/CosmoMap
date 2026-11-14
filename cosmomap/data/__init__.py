"""Data loading, normalisation and the patch-based ``tf.data`` pipeline."""

from cosmomap.data.container import CamelsData
from cosmomap.data.dataset import make_dataset, split_indices, split_simulations
from cosmomap.data.normalization import (
    denormalize_field,
    fit_normalization,
    is_log_field,
    load_normalization,
    normalize_field,
    save_normalization,
)
from cosmomap.data.synthetic import make_synthetic_camels

__all__ = [
    "CamelsData",
    "make_dataset",
    "split_indices",
    "split_simulations",
    "fit_normalization",
    "normalize_field",
    "denormalize_field",
    "save_normalization",
    "load_normalization",
    "is_log_field",
    "make_synthetic_camels",
]
