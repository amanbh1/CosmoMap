"""Shared settings for the CosmoMap rerun on CAMELS IllustrisTNG LH (128^3).

Key design choices:
* GLOBAL normalisation: log10 fields standardised with a single mean/std per
  field computed on training simulations only. No per-patch target statistics
  are used anywhere, so predictions are in absolute physical units.
* One fixed simulation-level split (seeded): 800 train / 100 val / 100 test,
  used for every model, baseline and metric.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

RAW = Path(os.path.expanduser("~/camels"))
PREP = Path(os.path.expanduser("~/camels/prep"))
RUNS = Path(os.path.expanduser("~/camels/runs"))

FIELDS_IN = ["Mcdm", "Vcdm"]
FIELDS_OUT = ["T", "ne"]
ALL_FIELDS = FIELDS_IN + FIELDS_OUT
REDSHIFTS = [0.0, 0.5, 1.0, 1.5, 2.0]
PARAM_NAMES = ["Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2"]
N_SIMS = 1000
RES = 128
BOX = 25.0  # Mpc/h

# Floors applied before log10 (well below the physical minima in CAMELS).
FLOORS = {"Mcdm": 1e6, "Vcdm": 1.0, "T": 10.0, "ne": 1e-12}


def raw_path(field: str, z: float) -> Path:
    return RAW / f"Grids_{field}_IllustrisTNG_LH_128_z={z:.1f}.npy"


def prep_path(field: str, z: float) -> Path:
    return PREP / f"{field}_z{z:.1f}.f16.npy"


def available_redshifts() -> list[float]:
    return [z for z in REDSHIFTS if all(prep_path(f, z).exists() for f in ALL_FIELDS)]


def split(seed: int = 2025) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_SIMS)
    return {"train": np.sort(perm[:800]), "val": np.sort(perm[800:900]), "test": np.sort(perm[900:])}


def load_params() -> np.ndarray:
    p = np.loadtxt(RAW / "params_LH_IllustrisTNG.txt", dtype=np.float32)
    return p[:, :6]


def load_stats() -> dict:
    return json.loads((PREP / "stats.json").read_text())


def cond_vector(params: np.ndarray, z: float, stats: dict, use: bool = True) -> np.ndarray:
    """Standardised [6 params, z] conditioning vector (zeros if conditioning disabled)."""
    pm, ps = np.array(stats["param_mean"]), np.array(stats["param_std"])
    v = np.concatenate([(params - pm) / ps, [(z - 1.0) / 0.7071]]).astype(np.float32)
    return v if use else np.zeros_like(v)


def to_log_physical(x_std: np.ndarray, field: str, stats: dict) -> np.ndarray:
    """Standardised -> log10 physical units."""
    return x_std * stats["std"][field] + stats["mean"][field]


# ---- Which trained runs the paper uses (main model = plain-MSE loss after the ablation study) ----
MAIN = "mse_z0_s0"                                  # CosmoMap, z=0, seed 0 (figures)
MAIN_SEEDS = ["mse_z0_s0", "mse_z0_s1", "mse_z0_s2"]
MAIN_ALLZ = "mse_allz_s0"                           # multi-redshift CosmoMap
NOFILM = "nofilm_mse_z0_s0"                         # unconditioned ablation (same loss)
SMALL = "small_mse_z0_s0"
HETERO = "film_z0_s0"                               # heteroscedastic-likelihood variant (CosmoMap-sigma)
HETERO_SEEDS = ["film_z0_s0", "film_z0_s1", "film_z0_s2"]
GBT = "gbt_full2k"                                  # tree baseline at z=0: all 800 training sims, early-stopped (converged)
BINNED = "binned_full"                               # binned lookup at z=0 from all 800 training sims
