# Method → code

A short guide from each part of the method to the code that implements it. The paper has
the full description and all results; this file only says where things live.

There are two places to look:

* `paper_experiments/`: the scripts that produced the results of the paper.
* `cosmomap/`: the installable package. Its model components are the ones the paper
  scripts import, and its small training pipeline follows the same method.

## Problem

Hydrodynamical simulations are expensive. CosmoMap learns a fast map from the dark
matter fields to the gas fields, conditioned on the cosmological and feedback
parameters of the simulation.

| Symbol | Meaning | Role |
|---|---|---|
| `Mcdm` | dark matter mass density | input channel (log₁₀) |
| `Vcdm` | dark matter velocity magnitude | input channel (log₁₀) |
| `T` | gas temperature (mass-weighted) | target (log₁₀) |
| `ne` | electron number density | target (log₁₀) |
| `Omega_m, sigma_8` | cosmological parameters | FiLM conditioning |
| `A_SN1, A_AGN1, A_SN2, A_AGN2` | supernova and AGN feedback parameters | FiLM conditioning |
| `z` | redshift | FiLM conditioning |

## Where each step is implemented

| Step | Paper scripts | Package |
|---|---|---|
| Normalisation: log₁₀ after a small floor, then one mean and standard deviation per field, fitted on training simulations only | `prep.py`, `common.py` | `cosmomap/data/normalization.py` |
| Fixed simulation-level split, 800 / 100 / 100 | `common.split` | `cosmomap/data/dataset.py` |
| Training patches: random periodic 64³ crops, 48 cube symmetries | `train.py` (`Sampler`) | `cosmomap/data/dataset.py` |
| FiLM layer, `h → (1 + γ(c)) ⊙ h + β(c)` | uses the package | `cosmomap/models/film.py` |
| Conditioning MLP and residual FiLM block | uses the package | `cosmomap/models/blocks.py` |
| U-Net, 64 → 128 → 256 channels, GroupNorm (8 groups), 11,056,772 parameters | `model.py` | `cosmomap/models/unet3d.py` |
| Losses: MSE (CosmoMap); composite likelihood loss (CosmoMap-σ); the ablation losses | `train.py` | `cosmomap/losses.py` |
| Optimiser: AdamW, one-epoch warm-up, cosine decay, gradient clipping, mixed precision | `train.py` | `cosmomap/train.py` |
| Inference: periodic 64³ windows, central 32³ kept; full-box periodic variant | `predict.py`, `model_periodic.py` | not in the package |
| Uncertainty: predicted variance plus Monte Carlo dropout | `predict.py` | `cosmomap/uncertainty.py` |
| Field metrics, density regimes, power spectra, luminosities | `metrics.py` | `cosmomap/evaluate.py`, `cosmomap/metrics.py` (simpler) |
| Baselines: binned lookup and gradient-boosted trees | `baselines.py` | not in the package |
| Scaling relations, bright structures, profiles, parameter-space robustness | `validation_extra.py`, `structure_purity.py` | `cosmomap/validation.py` (general utilities) |
| Parameter recovery from summary statistics | `param_recovery.py` | `cosmomap/inference.py` (a different feature set) |
| Calibration and rescaling of the uncertainties | `calib.py`, `pair_calib.py`, `regime_calib.py`, `env_recalib.py` | `cosmomap/uncertainty.py` (coverage only) |
| Gravity-only inputs: gridding the N-body velocities | `grid_velocity.py`, `kernels.py` | not in the package |
| Mock X-ray lightcones with noise and source detection | `lightcone.py` | `cosmomap/xray/` (a simple projected map) |

Where the package column says "simpler", "general utilities" or "a different feature
set", the analysis in the paper is the one in the paper scripts.

## Key equations

- **Normalisation:** `x' = (log10(max(X, floor)) − μ_X) / σ_X`, with `μ_X`, `σ_X` from the training set
- **FiLM:** `h → (1 + γ(c)) ⊙ h + β(c)`
- **CosmoMap loss:** mean squared error of the predicted means
- **CosmoMap-σ loss:** `0.408·NLL + 0.150·MSE + 0.134·MAE + 0.207·Huber + 0.01·⟨s²⟩`, with `NLL = ½⟨(y − μ)²/s² + log s²⟩`
- **X-ray emissivity (bremsstrahlung):** `ε_X = 1.42×10⁻²⁷ · n_e² · √T`
