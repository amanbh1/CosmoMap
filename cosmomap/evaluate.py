"""Evaluation: overall accuracy, accuracy stratified by local dark-matter density,
and uncertainty calibration. All metrics are in log10 of the physical fields
(RMSE and bias in dex), obtained with the training-set normalisation constants
saved by the training run.

Usage::

    cosmomap-evaluate --config outputs/synthetic/config.yaml \
                      --model outputs/synthetic/final_model.keras --synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

import cosmomap.models  # noqa: F401  -- import registers FiLM3D so saved models load
from cosmomap.config import Config
from cosmomap.data import (
    load_normalization,
    make_dataset,
    make_synthetic_camels,
    split_simulations,
)
from cosmomap.data.camels import load_camels
from cosmomap.data.normalization import Normalization
from cosmomap.metrics import pearson_r, r2_score, rmse
from cosmomap.train import NORMALIZATION_FILE
from cosmomap.uncertainty import calibration_coverage, mc_dropout_predict

#: Density regimes by percentile of local dark-matter density, pooled over the
#: evaluated patches.
PERCENTILE_EDGES = [0, 10, 30, 50, 80, 95, 100]
REGIME_LABELS = ["deep_void", "void", "sheet", "filament", "outskirt", "halo_core"]


def _collect(model, dataset, n_out):
    """Run the model over a finite dataset, returning stacked arrays."""
    y_true, y_mean, dm = [], [], []
    for inputs, targets in dataset:
        pred = tf.cast(model(inputs, training=False), tf.float32).numpy()
        y_mean.append(pred[..., :n_out])
        y_true.append(targets.numpy())
        dm.append(inputs["image"].numpy()[..., 0])  # standardised DM density channel
    return (
        np.concatenate(y_true, axis=0),
        np.concatenate(y_mean, axis=0),
        np.concatenate(dm, axis=0),
    )


def stratified_by_density(
    y_true: np.ndarray, y_pred: np.ndarray, dm_density: np.ndarray, field_names: list[str]
) -> dict:
    """R^2 and bias per field within each density regime."""
    thresholds = np.percentile(dm_density.ravel(), PERCENTILE_EDGES)
    results: dict[str, dict] = {}
    for i, label in enumerate(REGIME_LABELS):
        lo, hi = thresholds[i], thresholds[i + 1]
        mask = (dm_density >= lo) & (dm_density <= hi if i == len(REGIME_LABELS) - 1 else dm_density < hi)
        if not np.any(mask):
            continue
        regime: dict[str, dict] = {}
        for c, name in enumerate(field_names):
            yt = y_true[..., c][mask]
            yp = y_pred[..., c][mask]
            regime[name] = {
                "r2": r2_score(yt, yp),
                "bias": float(np.mean(yp - yt)),
                "n_voxels": int(mask.sum()),
            }
        results[label] = regime
    return results


def evaluate(
    config: Config, model: keras.Model, data, indices, stats: Normalization, mc_passes: int = 0
) -> dict:
    """Compute overall, density-stratified and (optionally) calibration metrics.

    ``stats`` are the normalisation constants the model was trained with.
    """
    n_out = config.n_out
    field_names = config.data.fields_out
    dataset = make_dataset(data, indices, config, training=False, stats=stats)
    y_true, y_mean, dm = _collect(model, dataset, n_out)
    # standardised -> log10 physical units, with constants fixed at training time
    scale = np.array([stats[f]["std"] for f in field_names], np.float32)
    shift = np.array([stats[f]["mean"] for f in field_names], np.float32)
    y_true, y_mean = y_true * scale + shift, y_mean * scale + shift

    overall = {
        name: {
            "r2": r2_score(y_true[..., c], y_mean[..., c]),
            "rmse": rmse(y_true[..., c], y_mean[..., c]),
            "pearson_r": pearson_r(y_true[..., c], y_mean[..., c]),
        }
        for c, name in enumerate(field_names)
    }

    report: dict = {
        "overall": overall,
        "by_density_regime": stratified_by_density(y_true, y_mean, dm, field_names),
    }

    if mc_passes > 0 and config.model.heteroscedastic and config.loss.w_nll > 0:
        # Calibration on a single representative batch (standardised units; coverage
        # is unchanged by the linear map to log10 physical units).
        inputs, targets = next(iter(dataset))
        uq = mc_dropout_predict(
            model, inputs, n_passes=mc_passes, n_out=n_out, heteroscedastic=True
        )
        report["calibration_95pct_coverage"] = {
            name: calibration_coverage(
                targets.numpy()[..., c], uq["mean"][..., c], uq["total"][..., c]
            )
            for c, name in enumerate(field_names)
        }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained CosmoMap model.")
    parser.add_argument("--config", required=True, help="path to the run's config.yaml")
    parser.add_argument("--model", required=True, help="path to a saved .keras model")
    parser.add_argument("--synthetic", action="store_true", help="evaluate on generated data")
    parser.add_argument("--n-sims", type=int, default=16)
    parser.add_argument("--mc-passes", type=int, default=20, help="MC-dropout passes (0 to skip)")
    parser.add_argument(
        "--stats", default=None, help="normalisation constants (default: next to the config)"
    )
    args = parser.parse_args(argv)

    config = Config.from_yaml(args.config)
    stats = load_normalization(args.stats or Path(args.config).parent / NORMALIZATION_FILE)
    model = keras.models.load_model(args.model, compile=False)
    data = (
        make_synthetic_camels(config, n_sims=args.n_sims, seed=config.train.seed)
        if args.synthetic
        else load_camels(config)
    )
    _, _, test_idx = split_simulations(
        data.sim_ids, config.data.train_val_test_split, config.train.seed
    )

    report = evaluate(config, model, data, test_idx, stats, mc_passes=args.mc_passes)
    out_path = Path(config.train.output_dir) / "evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
