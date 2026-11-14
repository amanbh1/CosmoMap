"""Shared pytest fixtures. Tests run on CPU with tiny tensors so the whole suite
finishes quickly."""

from __future__ import annotations

import numpy as np
import pytest

from cosmomap.config import Config
from cosmomap.data import fit_normalization


@pytest.fixture
def tiny_config() -> Config:
    """A minimal config: 16^3 boxes, 8^3 patches, a 2-level net."""
    c = Config()
    c.data.resolution = 16
    c.data.patch_size = 8
    c.data.redshifts = [0.0, 1.0]
    c.data.train_val_test_split = [0.6, 0.2, 0.2]
    c.model.base_filters = 8
    c.model.depth = 2
    c.model.groups = 2
    c.model.cond_hidden = 16
    c.train.batch_size = 2
    c.train.epochs = 1
    c.train.steps_per_epoch = None
    c.train.warmup_epochs = 0
    c.train.mixed_precision = False
    return c


@pytest.fixture
def fit_stats():
    """Fit normalisation constants on all samples of a (small, synthetic) dataset."""

    def _fit(data, config: Config):
        return fit_normalization(
            data.fields, np.arange(data.n_samples), config.data.log_floors, config.data.log_epsilon
        )

    return _fit
