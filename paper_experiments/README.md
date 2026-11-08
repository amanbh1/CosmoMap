# Paper experiments

Scripts that produce every result, table and figure in *CosmoMap: Predicting Hot Gas Temperature and Density from Dark Matter with a Conditioned 3D U-Net* (Bhutani & Lieu). They use the `FiLM3D` layer from the `cosmomap` package in this repository and the CAMELS IllustrisTNG LH 128³ grids of the Multifield Dataset.

Data and run directories are set in `common.py` (`~/camels`, `~/camels/prep`, `~/camels/runs` by default).

## Pipeline

| Step | Script |
|---|---|
| Normalise the grids, fixed 800/100/100 split | `prep.py` |
| Train a model (`--loss mse` for CosmoMap, composite loss for CosmoMap-σ; `--no-cond`, `--base`, `--groups`, `--norm batch`, `--seed`, `--zs` for the ablations) | `train.py` |
| Predict full test boxes by tiled inference | `predict.py` |
| Field, regime, scale and luminosity metrics | `metrics.py` |
| Binned-lookup and gradient-boosted-tree baselines | `baselines.py` |
| Scaling relations, bright structures, profiles, parameter-space robustness | `validation_extra.py`, `structure_purity.py`, `outskirts.py` |
| Uncertainty calibration (global, paired, per-environment) and luminosity debiasing | `calib.py`, `pair_calib.py`, `regime_calib.py`, `env_recalib.py`, `lum_debias.py` |
| Parameter recovery from summary statistics | `param_recovery.py`, `citable_params.py` |
| Mock X-ray lightcones and completeness/purity versus flux | `lightcone.py`, `lc_completeness.py`, `lightcone_products.py` |
| Gravity-only velocity grids (PCS deposit) and their validation | `grid_velocity.py`, `kernels.py`, `validate_kernels.py` |
| Inference-scheme comparison and timing | `compare_inference.py`, `time_inference.py` |
| Figures and quoted numbers | `make_figures.py`, `figs_extra.py`, `paper_numbers.py`, `summ_seeds.py`, `summ_seeds2.py`, `summ_bn.py` |

Example (one CosmoMap seed at z = 0):

```bash
python prep.py
python train.py --name mse_z0_s0 --seed 0 --zs 0.0 --loss mse
python predict.py --name mse_z0_s0 --mc 0
python metrics.py mse_z0_s0
```

These are research scripts, kept as they were run; they are excluded from the package's lint checks.
