from pathlib import Path

import pytest

from cosmomap.config import Config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_follow_the_paper():
    c = Config()
    assert c.data.patch_size == 64
    assert c.data.cosmo_params == ["Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2"]
    assert c.n_cond == 7  # six CAMELS parameters and the redshift
    assert c.n_in == 2 and c.n_out == 2
    assert c.out_channels == 4  # 2 means + 2 log-variances
    assert c.input_shape == (64, 64, 64, 2)
    # CosmoMap: mean squared error of the predicted means
    assert (c.loss.w_mse, c.loss.w_nll, c.loss.w_mae, c.loss.w_huber, c.loss.lambda_variance) == (1, 0, 0, 0, 0)
    assert (c.train.batch_size, c.train.epochs, c.train.steps_per_epoch) == (4, 30, 1000)
    assert c.train.learning_rate == 2e-4 and c.train.weight_decay == 1e-5
    assert c.data.log_floors == {"Mcdm": 1e6, "Vcdm": 1.0, "T": 10.0, "ne": 1e-12}


def test_default_yaml_equals_defaults():
    """configs/default.yaml spells out the paper configuration; it must not drift from the code."""
    loaded, default = Config.from_yaml(CONFIGS / "default.yaml"), Config()
    assert loaded.model == default.model
    assert loaded.loss == default.loss
    for name in ("epochs", "steps_per_epoch", "batch_size", "learning_rate", "lr_schedule", "warmup_epochs",
                 "weight_decay", "grad_clip_norm", "mixed_precision"):
        assert getattr(loaded.train, name) == getattr(default.train, name), name
    for name in ("fields_in", "fields_out", "cosmo_params", "patch_size", "include_redshift_in_cond", "log_floors"):
        assert getattr(loaded.data, name) == getattr(default.data, name), name


def test_sigma_yaml_is_the_composite_loss():
    c = Config.from_yaml(CONFIGS / "sigma.yaml")
    assert (c.loss.w_nll, c.loss.w_mse, c.loss.w_mae, c.loss.w_huber, c.loss.lambda_variance) == (
        0.408, 0.150, 0.134, 0.207, 0.01,
    )
    assert c.model == Config().model  # same network as CosmoMap


@pytest.mark.parametrize("name", ["default.yaml", "sigma.yaml", "synthetic.yaml"])
def test_shipped_configs_load(name):
    c = Config.from_yaml(CONFIGS / name)
    assert c.data.patch_size % (2**c.model.depth) == 0


def test_yaml_roundtrip(tmp_path):
    c = Config()
    path = tmp_path / "config.yaml"
    c.to_yaml(path)
    loaded = Config.from_yaml(path)
    assert loaded == c


def test_unknown_keys_ignored():
    c = Config.from_dict({"data": {"patch_size": 32, "nonsense": 1}, "extra": True})
    assert c.data.patch_size == 32
