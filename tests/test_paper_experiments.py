# ruff: noqa: E402, I001
"""Tests of the scripts in ``paper_experiments/`` that produced the results of the paper.

They need no CAMELS data. They check that the network is the one the paper describes and
that the numerical building blocks behave as stated: the fixed split, the conditioning
vector, the tiled and full-box inference, the metrics, the losses, the gridding kernels,
the lightcone sampling and the profile fits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_experiments"))

import common
import kernels
import lightcone as pe_lightcone
import metrics as pe_metrics
import model as pe_model
import model_periodic
import predict
import train as pe_train
import validation_extra

from cosmomap.config import Config
from cosmomap.models import build_model

#: Table 1 of the paper.
PAPER_PARAMS = {
    "Decoder": 5_671_104,
    "Bottleneck": 3_540_480,
    "Encoder": 1_259_200,
    "FiLM layers": 543_488,
    "Conditioning MLP": 38_592,
    "Input stem": 3_648,
    "Output head": 260,
}


def _component(name: str) -> str:
    if name.startswith("cond_dense"):
        return "Conditioning MLP"
    if "_film" in name:
        return "FiLM layers"
    for prefix, group in (
        ("stem", "Input stem"),
        ("enc", "Encoder"),
        ("bottleneck", "Bottleneck"),
        ("dec", "Decoder"),
    ):
        if name.startswith(prefix):
            return group
    return "Output head"


# --------------------------------------------------------------------------- #
# Data handling
# --------------------------------------------------------------------------- #
def test_split_is_fixed_and_disjoint():
    a, b = common.split(), common.split()
    assert [len(a[k]) for k in ("train", "val", "test")] == [800, 100, 100]
    assert all(np.array_equal(a[k], b[k]) for k in a)  # same split on every call
    assert np.array_equal(np.sort(np.concatenate(list(a.values()))), np.arange(1000))


def test_conditioning_vector():
    stats = {"param_mean": [0.3, 0.8, 1.0, 1.0, 1.0, 1.0], "param_std": [0.1, 0.1, 1.0, 1.0, 0.5, 0.5]}
    params = np.array([0.4, 0.7, 2.0, 1.0, 1.5, 0.5], np.float32)
    c = common.cond_vector(params, 1.0, stats)
    assert c.shape == (7,)
    assert np.allclose(c, [1.0, -1.0, 1.0, 0.0, 1.0, -1.0, 0.0], atol=1e-5)
    assert not common.cond_vector(params, 1.0, stats, use=False).any()  # unconditioned network


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def test_network_matches_table_1():
    net = pe_model.build()
    assert net.count_params() == 11_056_772
    groups: dict[str, int] = {}
    for layer in net.layers:
        if layer.count_params():
            groups[_component(layer.name)] = groups.get(_component(layer.name), 0) + layer.count_params()
    assert groups == PAPER_PARAMS


def test_small_network_size():
    small = pe_model.build(base=20, groups=4).count_params()
    assert 1.15e6 < small < 1.25e6  # "1.2e6 parameters", nine times fewer than CosmoMap
    assert round(11_056_772 / small) == 9


def test_package_model_is_the_paper_network():
    config = Config()
    config.data.patch_size = 64
    config.data.include_redshift_in_cond = True
    assert build_model(config).count_params() == 11_056_772


def test_network_is_fully_convolutional():
    net = pe_model.build(base=8, groups=2, cond_hidden=16)
    for size in (16, 24):
        x = {"image": tf.zeros((1, size, size, size, 2)), "cosmo": tf.zeros((1, 7))}
        assert tuple(net(x, training=False).shape) == (1, size, size, size, 4)


def test_full_box_model_is_periodic():
    """The periodic model takes the standard model's weights unchanged and commutes with
    periodic shifts of the box (by multiples of the total pooling factor, 8).

    In float32 the shifted outputs agree to ~1e-3: the randomly initialised network amplifies
    rounding differences (a 1e-7 relative perturbation of the input moves the output by as
    much), whereas zero padding gives differences of order 1. The tolerance sits between."""
    tf.keras.utils.set_random_seed(0)
    kw = dict(base=8, groups=2, cond_hidden=16)
    src = pe_model.build(**kw)
    net = model_periodic.copy_weights(src, model_periodic.build_periodic(**kw))
    for a, b in zip(src.get_weights(), net.get_weights(), strict=True):
        assert np.array_equal(a, b)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 16, 16, 16, 2)).astype(np.float32)
    c = rng.standard_normal((1, 7)).astype(np.float32)
    shift = (8, 0, 8)
    y = net({"image": x, "cosmo": c}, training=False).numpy()
    y_shifted = net({"image": np.roll(x, shift, (1, 2, 3)), "cosmo": c}, training=False).numpy()
    assert np.allclose(np.roll(y, shift, (1, 2, 3)), y_shifted, atol=1e-2)
    # zero padding breaks this symmetry at the box faces, which is why tiling needs margins
    z = src({"image": x, "cosmo": c}, training=False).numpy()
    z_shifted = src({"image": np.roll(x, shift, (1, 2, 3)), "cosmo": c}, training=False).numpy()
    assert np.abs(np.roll(z, shift, (1, 2, 3)) - z_shifted).max() > 0.1


# --------------------------------------------------------------------------- #
# Tiled inference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("core", "margin"), [(32, 16), (64, 32)])
def test_tiling_reassembles_the_box(monkeypatch, core, margin):
    monkeypatch.setattr(predict, "C", core)
    monkeypatch.setattr(predict, "M", margin)
    rng = np.random.default_rng(0)
    box = rng.standard_normal((128, 128, 128, 1)).astype(np.float32)
    w, starts = predict.windows(box)
    assert w.shape == ((128 // core) ** 3, core + 2 * margin, core + 2 * margin, core + 2 * margin, 1)
    assert np.array_equal(predict.assemble(w, starts, 1), box)  # cores tile the box exactly once
    # the first window is the periodic neighbourhood of the first core
    assert np.array_equal(w[0], np.roll(box, (margin, margin, margin), (0, 1, 2))[: core + 2 * margin, : core + 2 * margin, : core + 2 * margin])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_power_spectrum_statistics_of_identical_fields():
    rng = np.random.default_rng(0)
    a = np.exp(0.3 * rng.standard_normal((128, 128, 128)))
    p_true, p_pred, p_cross = pe_metrics.pk_cross(a, a)
    filled = p_true > 0
    assert filled[1:].all()  # every bin above the fundamental holds modes
    assert np.allclose(np.sqrt(p_pred[filled] / p_true[filled]), 1.0)  # transfer function
    assert np.allclose(p_cross[filled] / np.sqrt(p_true[filled] * p_pred[filled]), 1.0)  # r(k)
    # half the amplitude of the overdensity: transfer function 0.5, still perfectly correlated
    b = a.mean() * (1 + 0.5 * (a / a.mean() - 1))
    p_true, p_pred, p_cross = pe_metrics.pk_cross(a, b)
    assert np.allclose(np.sqrt(p_pred[filled] / p_true[filled]), 0.5)
    assert np.allclose(p_cross[filled] / np.sqrt(p_true[filled] * p_pred[filled]), 1.0)


def test_field_metrics_for_a_constant_offset():
    rng = np.random.default_rng(1)
    t = {"T": 4 + rng.standard_normal((128, 128, 128)).astype(np.float32), "ne": -7 + 0.5 * rng.standard_normal((128, 128, 128)).astype(np.float32)}
    p = {"T": t["T"] + 0.1, "ne": t["ne"] - 0.05}
    rho = rng.standard_normal(t["T"].shape).astype(np.float32)
    pct = np.argsort(np.argsort(rho.ravel())).reshape(rho.shape) / rho.size * 100
    rec = pe_metrics.score_one(t, p, None, pct)
    assert rec["T"]["bias"] == pytest.approx(0.1, abs=1e-5)
    assert rec["T"]["rmse"] == pytest.approx(0.1, abs=1e-5)
    assert rec["T"]["r2"] == pytest.approx(1 - 0.01 / float(np.var(t["T"])), abs=1e-4)
    for name, _, _ in pe_metrics.REGIMES:  # the offset is the same in every density regime
        assert rec["ne"][f"{name}_bias"] == pytest.approx(-0.05, abs=1e-5)
    # emissivity ~ ne^2 T^(1/2): the luminosity ratio is 10^(2 * -0.05 + 0.5 * 0.1)
    assert rec["eps"]["L_ratio"] == pytest.approx(10 ** (-0.05), rel=1e-4)
    # the six regimes partition the voxels
    counts = [((pct >= lo) & (pct < hi) if hi < 100 else pct >= lo).sum() for _, lo, hi in pe_metrics.REGIMES]
    assert sum(counts) == pct.size


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def test_losses_match_the_paper():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((2, 4, 4, 4, 2)).astype(np.float32)
    mu = rng.standard_normal(y.shape).astype(np.float32)
    lv = (0.5 * rng.standard_normal(y.shape)).astype(np.float32)
    pred = np.concatenate([mu, lv], -1)
    e = y - mu
    mse, mae = np.mean(e**2), np.mean(np.abs(e))
    huber = np.mean(np.where(np.abs(e) < 1, 0.5 * e**2, np.abs(e) - 0.5))
    nll = 0.5 * np.mean(e**2 * np.exp(-lv) + lv)

    assert float(pe_train.loss_mean_only(y, pred, mse_only=True)) == pytest.approx(mse, rel=1e-5)  # CosmoMap
    composite = 0.408 * nll + 0.150 * mse + 0.134 * mae + 0.207 * huber + 0.01 * np.mean(np.exp(lv))
    assert float(pe_train.loss_fn(y, pred)) == pytest.approx(composite, rel=1e-5)  # CosmoMap-sigma
    assert float(pe_train.loss_mean_only(y, pred)) == pytest.approx(0.150 * mse + 0.134 * mae + 0.207 * huber, rel=1e-5)
    assert float(pe_train.loss_nll(y, pred)) == pytest.approx(nll, rel=1e-5)
    beta_nll = np.mean(np.exp(0.5 * lv) * 0.5 * (e**2 * np.exp(-lv) + lv))  # weight s^(2 beta), beta = 0.5
    assert float(pe_train.loss_nll(y, pred, beta=0.5)) == pytest.approx(beta_nll, rel=1e-5)


# --------------------------------------------------------------------------- #
# Velocity gridding, lightcone sampling, profiles
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order", [3, 4])
def test_gridding_kernels_conserve_mass(order):
    rng = np.random.default_rng(0)
    n = 4000
    pos = rng.uniform(0, kernels.L, (n, 3))
    mean, mass = kernels.deposit_kernel(pos, np.full(n, 7.0), order)
    assert mass.sum() == pytest.approx(n, rel=1e-6)  # the weights of each particle sum to one
    assert mass.min() >= 0
    assert np.allclose(mean[mass > 1e-6], 7.0)  # mass-weighted mean of a constant is the constant


def test_lightcone_sampling_is_bilinear_and_periodic():
    rng = np.random.default_rng(0)
    img = rng.standard_normal((pe_lightcone.RES, pe_lightcone.RES))
    cell = pe_lightcone.BOX / pe_lightcone.RES
    i, j = np.array([0, 5, 127]), np.array([3, 127, 0])
    assert np.allclose(pe_lightcone.sample_periodic(img, i * cell, j * cell), img[i, j])
    assert np.allclose(pe_lightcone.sample_periodic(img, i * cell + pe_lightcone.BOX, j * cell - 2 * pe_lightcone.BOX), img[i, j])
    mid = pe_lightcone.sample_periodic(img, np.array([10.5 * cell]), np.array([20.0 * cell]))
    assert np.allclose(mid, 0.5 * (img[10, 20] + img[11, 20]))


def test_beta_model_fit_and_profile():
    r = validation_extra.R_MID
    truth = (1.3, 0.3, 0.65)  # log S0, r_c, beta
    fit = validation_extra.fit_beta(r, 10 ** validation_extra.beta_model(r, *truth))
    assert fit[2] == pytest.approx(truth[2], abs=0.01)
    assert fit[1] == pytest.approx(truth[1], abs=0.02)
    side = 2 * validation_extra.HALF + 1
    assert np.allclose(validation_extra.profile(np.ones((side, side, side))), side)  # flat cube: flat profile
