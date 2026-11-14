import numpy as np
import pytest
import tensorflow as tf

from cosmomap.config import Config
from cosmomap.losses import CosmoMapLoss, make_loss
from cosmomap.metrics import power_spectrum_3d, r2_score, rmse, ssim3d

CONFIGS = __import__("pathlib").Path(__file__).resolve().parents[1] / "configs"


def _batch(seed=0, shape=(2, 4, 4, 4, 2)):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(shape).astype(np.float32)
    mu = rng.standard_normal(shape).astype(np.float32)
    log_var = (0.5 * rng.standard_normal(shape)).astype(np.float32)
    return y, mu, log_var


def test_loss_finite_and_nonnegative(tiny_config):
    loss = make_loss(tiny_config)
    p, n = tiny_config.data.patch_size, tiny_config.n_out
    y_true = tf.random.normal((2, p, p, p, n))
    y_pred = tf.random.normal((2, p, p, p, tiny_config.out_channels))
    value = float(loss(y_true, y_pred).numpy())
    assert np.isfinite(value) and value >= 0.0


def test_loss_prefers_correct_mean(tiny_config):
    loss = make_loss(tiny_config)
    p, n = tiny_config.data.patch_size, tiny_config.n_out
    y_true = tf.random.normal((2, p, p, p, n))
    zeros_logvar = tf.zeros_like(y_true)
    good = tf.concat([y_true, zeros_logvar], axis=-1)
    bad = tf.concat([y_true + 1.0, zeros_logvar], axis=-1)
    assert float(loss(y_true, good).numpy()) < float(loss(y_true, bad).numpy())


def test_default_loss_is_mean_squared_error():
    """CosmoMap: MSE of the predicted means; the variance channels do not enter."""
    y, mu, log_var = _batch()
    loss = make_loss(Config())
    value = float(loss(y, np.concatenate([mu, log_var], -1)))
    assert value == pytest.approx(float(np.mean((y - mu) ** 2)), rel=1e-5)
    assert float(loss(y, np.concatenate([mu, log_var + 3.0], -1))) == pytest.approx(value, rel=1e-6)


def test_composite_loss_matches_the_paper():
    """CosmoMap-sigma: 0.408 NLL + 0.150 MSE + 0.134 MAE + 0.207 Huber + 0.01 <s^2>."""
    y, mu, log_var = _batch(1)
    e = y - mu
    nll = 0.5 * np.mean(e**2 * np.exp(-log_var) + log_var)
    huber = np.mean(np.where(np.abs(e) < 1, 0.5 * e**2, np.abs(e) - 0.5))
    expected = (
        0.408 * nll + 0.150 * np.mean(e**2) + 0.134 * np.mean(np.abs(e)) + 0.207 * huber
        + 0.01 * np.mean(np.exp(log_var))
    )
    loss = make_loss(Config.from_yaml(CONFIGS / "sigma.yaml"))
    assert float(loss(y, np.concatenate([mu, log_var], -1))) == pytest.approx(float(expected), rel=1e-5)


def test_loss_without_variance_head():
    y, mu, _ = _batch(2)
    loss = CosmoMapLoss(n_out=2, heteroscedastic=False, w_nll=0.408, lambda_variance=0.01)
    assert float(loss(y, mu)) == pytest.approx(float(np.mean((y - mu) ** 2)), rel=1e-5)  # likelihood terms skipped


def test_loss_serialisation_roundtrip():
    loss = make_loss(Config.from_yaml(CONFIGS / "sigma.yaml"))
    clone = CosmoMapLoss.from_config(loss.get_config())
    y, mu, log_var = _batch(3)
    pred = np.concatenate([mu, log_var], -1)
    assert float(clone(y, pred)) == pytest.approx(float(loss(y, pred)), rel=1e-6)


def test_metrics_on_perfect_prediction():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((4, 8, 8)).astype(np.float32)
    assert r2_score(y, y) > 0.999
    assert rmse(y, y) < 1e-6
    assert ssim3d(y, y) > 0.999


def test_power_spectrum_runs():
    rng = np.random.default_rng(0)
    k, pk = power_spectrum_3d(rng.standard_normal((16, 16, 16)))
    assert k.shape == pk.shape
    assert np.all(np.isfinite(pk))
