# Paper experiments

The scripts that produced every result, table and figure in *CosmoMap: Predicting Hot Gas
Temperature and Density from Dark Matter with a Conditioned 3D U-Net* (Bhutani & Lieu).
They import the FiLM layer and residual blocks from the `cosmomap` package in this
repository and read the 128³ grids of the CAMELS Multifield Dataset (IllustrisTNG LH set).

These are research scripts, kept close to the form in which they were run. They are
excluded from the package's lint checks; their numerical building blocks are covered by
`tests/test_paper_experiments.py`.

## Setup

```bash
pip install -e "..[paper]"       # from this directory; installs the package and the extra dependencies
bash download.sh                 # CAMELS grids -> ~/camels  (ZS="0.0 0.5 1.0 1.5 2.0" for all redshifts)
```

Run every script from this directory. Data and results live under `~/camels` (set in
`common.py`):

| Path | Contents |
|---|---|
| `~/camels/Grids_<field>_IllustrisTNG_LH_128_z=<z>.npy` | raw grids, `<field>` = `Mcdm`, `Vcdm`, `T`, `ne` (8.4 GB each) |
| `~/camels/params_LH_IllustrisTNG.txt` | the six parameters of each simulation |
| `~/camels/prep/` | normalised float16 copies and `stats.json`, the normalisation constants (written by `prep.py`) |
| `~/camels/runs/<name>/` | one directory per model or baseline: weights, `config.json`, `history.json`, predictions in `pred/` and `pred_val/`, `metrics.json` |
| `~/camels/runs/*.json` | results of the analysis scripts |

The paper's runs used one NVIDIA RTX 4070 (12 GB), Python 3.12 and TensorFlow 2.21.
One training run takes about three hours, predicting the 100 test boxes about four
minutes, and `metrics.py` one to two minutes per run.

## Pipeline

| Step | Script |
|---|---|
| Normalise the grids; constants from training simulations at z = 0; fixed 800/100/100 split | `prep.py`, `common.py` |
| Train a model | `train.py` |
| Predict full boxes by tiled inference (64³ windows, central 32³ kept), or in one periodic pass | `predict.py`, `model_periodic.py` |
| Field, regime, scale and luminosity metrics | `metrics.py` |
| Binned-lookup and gradient-boosted-tree baselines | `baselines.py` |
| Scaling relations, bright structures, profiles, parameter-space robustness | `validation_extra.py`, `structure_purity.py`, `outskirts.py` |
| Uncertainty calibration (global, paired, per environment) and luminosity debiasing | `calib.py`, `pair_calib.py`, `regime_calib.py`, `env_recalib.py`, `lum_debias.py` |
| Parameter recovery from summary statistics | `param_recovery.py`, `citable_params.py` |
| Mock X-ray lightcones; completeness and purity against flux; FITS products | `lightcone.py`, `lc_completeness.py`, `lightcone_products.py` |
| Gravity-only velocity grids (PCS deposit) and their validation | `grid_velocity.py`, `kernels.py`, `validate_kernels.py` |
| Inference-scheme comparison and timing | `compare_inference.py`, `time_inference.py` |
| Quoted numbers | `paper_numbers.py`, `quoted_numbers.py`, `summ_seeds.py`, `summ_seeds2.py`, `summ_bn.py` |
| Figures (written to `./figures/`) | `make_figures.py`, `figs_extra.py` |

## Models

All networks are trained at z = 0 unless stated; `<s>` is the seed (0, 1, 2).

| Model in the paper | Run name | Training command |
|---|---|---|
| CosmoMap | `mse_z0_s<s>` | `python train.py --name mse_z0_s<s> --seed <s> --zs 0.0 --loss mse` |
| CosmoMap-σ | `film_z0_s<s>` | `python train.py --name film_z0_s<s> --seed <s> --zs 0.0` |
| U-Net, no conditioning | `nofilm_mse_z0_s<s>` | `... --loss mse --no-cond` |
| Small network | `small_mse_z0_s<s>` | `... --loss mse --base 20 --groups 4` |
| MSE + MAE + Huber loss | `abl_nounc_z0` | `... --seed 0 --loss nounc` |
| Pure Gaussian likelihood | `nll_z0_s0` | `... --seed 0 --loss nll` |
| β-NLL | `betanll_z0_s0_last` | `... --name betanll_z0_s0 --seed 0 --loss betanll`, then take the last-epoch weights (below) |
| CosmoMap, batch normalisation | `bn_mse_z0_s0` | `... --seed 0 --loss mse --norm batch` |
| Multi-redshift CosmoMap | `mse_allz_s0` | `python train.py --name mse_allz_s0 --seed 0 --loss mse` (all prepared redshifts) |

Then, for each run,

```bash
python predict.py --name <run> --mc 0        # --mc 8 for the likelihood-trained models (Monte Carlo dropout)
python metrics.py <run>
```

β-NLL is the one model the paper evaluates at the last epoch instead of the checkpoint with the
lowest validation loss, because its validation loss does not track accuracy (Section 6.2 of the
paper). `train.py` saves both `best.weights.h5` and `last.weights.h5`; to score the last epoch,
put its weights in a run directory of their own:

```bash
mkdir ~/camels/runs/betanll_z0_s0_last
cp ~/camels/runs/betanll_z0_s0/config.json ~/camels/runs/betanll_z0_s0_last/
cp ~/camels/runs/betanll_z0_s0/last.weights.h5 ~/camels/runs/betanll_z0_s0_last/best.weights.h5
```

`predict.py --split val` writes validation predictions (`pred_val/`), which the calibration
scripts need for the CosmoMap-σ and CosmoMap seeds and `param_recovery.py` needs for
`mse_z0_s0` and `nofilm_mse_z0_s0`.

Baselines (they write `binned<tag>` and `gbt<tag>`):

```bash
python baselines.py --zs 0.0 --nsim 800 --iters 2000 --tag _full2k   # trees of the main table
python baselines.py --zs 0.0 --nsim 800 --tag _full                  # binned lookup of the main table; trees capped at 400 iterations
python baselines.py --zs 0.0                                         # 160 simulations (convergence check)
python baselines.py --tag _allz                                      # all redshifts, 160 simulations (redshift table)
```

Other inputs and inference schemes are written as separate runs with `--as-run`:

```bash
# gravity-only inputs (bash download.sh nbody; python grid_velocity.py nbody)
python predict.py --name mse_z0_s0 --mc 0 --nbody ~/camels/Grids_Mtot_Nbody_IllustrisTNG_LH_128_z=0.0.npy --as-run test_nbody
python predict.py --name mse_z0_s0 --mc 0 --nbody ~/camels/Grids_Mtot_Nbody_IllustrisTNG_LH_128_z=0.0.npy \
                  --nbody-vel ~/camels/Grids_Vcdm_Nbody_test_pcs_z=0.0.npy --as-run test_nbody_full
# inference schemes: see the header of compare_inference.py
python predict.py --name mse_z0_s0 --mc 0 --periodic --as-run test_periodic
```

## Analyses

After the runs above exist (`common.py` names the ones each script reads):

```bash
python calib.py film_z0_s0 0.0          # likewise for the other likelihood-trained runs
python validation_extra.py && python structure_purity.py
python param_recovery.py && python citable_params.py
python pair_calib.py && python regime_calib.py && python env_recalib.py   # env_recalib.py takes about half an hour
python lum_debias.py film_z0_s0 0.0
python lightcone.py 10 && python lc_completeness.py && python lightcone_products.py
python paper_numbers.py && python quoted_numbers.py && python compare_inference.py
python make_figures.py && python figs_extra.py
```

## Reproducibility

The split, the samplers and the analyses are seeded. Training and inference on a GPU are
not bit-reproducible: re-predicting from the same weights changes a per cent or two of the
voxels by one or two float16 steps (below 0.01 dex) and leaves the quoted metrics unchanged at the
precision given in the paper, and a retrained model differs by about the seed-to-seed
scatter quoted there.
