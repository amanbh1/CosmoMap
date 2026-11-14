"""End-to-end smoke tests on synthetic data: training, evaluation and the mock-map
script must run and use only the normalisation constants saved by training."""

import importlib.util
import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from cosmomap.data import load_normalization, make_synthetic_camels, split_simulations
from cosmomap.evaluate import evaluate
from cosmomap.train import NORMALIZATION_FILE, build_optimizer, learning_rate_schedule, train

ROOT = Path(__file__).resolve().parents[1]


def test_train_evaluate_and_map(tiny_config, tmp_path):
    out = tmp_path / "out"
    tiny_config.train.output_dir = str(out)
    tiny_config.train.epochs = 1
    tiny_config.loss.w_nll, tiny_config.loss.lambda_variance = 0.408, 0.01  # exercise the variance head
    model, history = train(tiny_config, synthetic=True, n_sims=8)
    assert "loss" in history.history and np.isfinite(history.history["loss"]).all()
    assert (out / "final_model.keras").exists()
    assert (out / "config.yaml").exists()

    # normalisation constants: saved with the run and fitted on the training simulations only
    stats = load_normalization(out / NORMALIZATION_FILE)
    assert set(stats) == {"Mcdm", "Vcdm", "T", "ne"}
    data = make_synthetic_camels(tiny_config, n_sims=8, seed=tiny_config.train.seed)
    train_idx, _, test_idx = split_simulations(data.sim_ids, tiny_config.data.train_val_test_split, tiny_config.train.seed)
    train_logT = np.log10(data.fields["T"][train_idx])
    assert abs(stats["T"]["mean"] - float(train_logT.mean())) < 1e-3

    report = evaluate(tiny_config, model, data, test_idx, stats, mc_passes=2)
    for name in ("T", "ne"):
        assert np.isfinite(report["overall"][name]["rmse"])  # dex
    assert set(report["calibration_95pct_coverage"]) == {"T", "ne"}
    json.dumps(report)  # serialisable

    # the mock-map script reads the same constants and never the true gas fields
    spec = importlib.util.spec_from_file_location("make_lightcone", ROOT / "scripts" / "make_lightcone.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    truth = {f: data.fields[f].copy() for f in ("T", "ne")}
    ne_a, t_a = script.predict_volumes(model, data, tiny_config, stats)
    data.fields["T"][:] = 1.0  # wipe the targets: the prediction must not change
    data.fields["ne"][:] = 1.0
    ne_b, t_b = script.predict_volumes(model, data, tiny_config, stats)
    for z in tiny_config.data.redshifts:
        assert np.array_equal(ne_a[z], ne_b[z]) and np.array_equal(t_a[z], t_b[z])
        assert (ne_a[z] > 0).all() and (t_a[z] > 0).all()
    data.fields.update(truth)


def test_learning_rate_schedule(tiny_config):
    """One epoch of linear warm-up to the peak rate, then a cosine decay to 2 per cent of it."""
    tiny_config.train.epochs, tiny_config.train.warmup_epochs, tiny_config.train.learning_rate = 30, 1, 2e-4
    assert isinstance(build_optimizer(tiny_config, steps_per_epoch=1000), keras.optimizers.AdamW)
    sched = learning_rate_schedule(tiny_config, steps_per_epoch=1000)
    assert float(sched(0)) == np.float32(2e-5)
    assert abs(float(sched(1000)) - 2e-4) < 1e-9
    assert abs(float(sched(30_000)) - 0.02 * 2e-4) < 1e-9
    assert float(sched(500)) < float(sched(1000)) > float(sched(15_000)) > float(sched(29_000))
    tiny_config.train.lr_schedule = "constant"
    assert learning_rate_schedule(tiny_config, steps_per_epoch=1000) == 2e-4
