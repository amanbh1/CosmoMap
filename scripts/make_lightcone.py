"""Generate a simple mock X-ray surface-brightness map from a trained model.

This is a small demonstration of the package's X-ray utilities on one simulation.
The lightcones analysed in the paper are built by ``paper_experiments/lightcone.py``.

Example
-------
    python scripts/make_lightcone.py \
        --config outputs/synthetic/config.yaml \
        --model  outputs/synthetic/final_model.keras \
        --synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

import cosmomap.models  # noqa: F401  -- import registers FiLM3D so saved models load
from cosmomap.config import Config
from cosmomap.data import load_normalization, make_synthetic_camels
from cosmomap.data.camels import load_camels
from cosmomap.data.normalization import Normalization, denormalize_field, normalize_field
from cosmomap.train import NORMALIZATION_FILE
from cosmomap.xray import LightconeConfig, build_lightcone, save_fits


def _center_slice(resolution: int, patch: int) -> tuple[slice, slice, slice]:
    s = (resolution - patch) // 2
    return (slice(s, s + patch),) * 3  # type: ignore[return-value]


def predict_volumes(model, data, config: Config, stats: Normalization, sim: int = 0):
    """Predict (n_e, T) volumes for one simulation across all redshifts.

    Inputs are normalised, and predictions returned to physical units, with the
    training-set constants in ``stats``; the true gas fields are never read.
    """
    patch = config.data.patch_size
    sl = _center_slice(data.resolution, patch)
    n_z = len(config.data.redshifts)
    n_out = config.n_out

    ne_vol: dict[float, np.ndarray] = {}
    t_vol: dict[float, np.ndarray] = {}
    for zi, z in enumerate(config.data.redshifts):
        idx = sim * n_z + zi
        x = np.stack(
            [normalize_field(data.fields[f][idx][sl], stats[f]) for f in config.data.fields_in],
            axis=-1,
        )[None]
        cond = data.cond[idx][None].astype("float32")
        pred = np.asarray(model({"image": x, "cosmo": cond}, training=False))[0, ..., :n_out]

        for ci, f in enumerate(config.data.fields_out):
            physical = denormalize_field(pred[..., ci], stats[f])
            if f == "ne":
                ne_vol[z] = physical
            elif f == "T":
                t_vol[z] = physical
    return ne_vol, t_vol


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-sims", type=int, default=8)
    parser.add_argument("--out", default=None)
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

    ne_vol, t_vol = predict_volumes(model, data, config, stats)
    lc = LightconeConfig(
        redshifts=config.data.redshifts, box_size_mpc_h=config.data.box_size_mpc_h, npix=128
    )
    sb = build_lightcone(ne_vol, t_vol, lc)

    out_path = Path(args.out or (Path(config.train.output_dir) / "lightcone.fits"))
    save_fits(sb, out_path, lc)
    print(f"Surface-brightness map: shape={sb.shape}, total flux={sb.sum():.3e}")
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
