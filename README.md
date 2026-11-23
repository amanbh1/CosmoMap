# CosmoMap

**Predicting hot gas temperature and density from dark matter with a conditioned 3D U-Net.**

Code for the paper *CosmoMap: Predicting Hot Gas Temperature and Density from Dark Matter
with a Conditioned 3D U-Net* (Aman Bhutani and Maggie Lieu).

CosmoMap is a 3D U-Net that maps the dark matter density and velocity magnitude of a
cosmological simulation to its gas temperature and electron number density. It is trained
on the IllustrisTNG Latin-hypercube set of [CAMELS](https://camels.readthedocs.io/) and
conditioned, through FiLM layers, on the two cosmological and four feedback parameters and
the redshift, so one network covers the whole parameter space. A full 128³ box
(25 h⁻¹ Mpc) is predicted in about 2 s on a single consumer GPU.

```
 log ρ_DM   ┐
            ├─► [ 3D FiLM U-Net, 11.1 M parameters ] ─► log T, log n_e  (and per-voxel variance for CosmoMap-σ)
 log |v_DM| ┘                 ▲
   Ω_m, σ_8, A_SN1, A_AGN1, A_SN2, A_AGN2, z
```

![Slices through a test box: inputs, simulation, CosmoMap, gradient-boosted trees and residuals](docs/figures/slices_z0.png)

*A 1.56 h⁻¹ Mpc slab through the test box with the median temperature R², at z = 0. Rows:
temperature, electron density and X-ray emissivity. Columns: inputs, simulation, CosmoMap,
gradient-boosted trees, and the residuals of each.*

## Results

Test-set accuracy at z = 0 (100 held-out simulations, full 128³ volumes, log₁₀ of the
physical fields). Fields are normalised with constants fixed from the training set, so no
statistic of a target field is used at test time.

| Model | log T: R² | RMSE [dex] | log n_e: R² | RMSE [dex] |
|---|---:|---:|---:|---:|
| Binned ρ_DM lookup | 0.173 | 0.671 | 0.652 | 0.307 |
| Gradient-boosted trees | 0.603 | 0.470 | 0.894 | 0.182 |
| U-Net, no conditioning | 0.647 | 0.438 | 0.906 | 0.172 |
| CosmoMap-σ (likelihood loss, with uncertainties) | 0.733 | 0.385 | 0.922 | 0.156 |
| **CosmoMap** | **0.765** | **0.362** | **0.934** | **0.142** |

The paper also covers the accuracy by density regime and by scale, redshifts up to z = 2,
an ablation study, bright emission structures and surface-brightness profiles, recovery of
cosmological parameters, uncertainty calibration, gravity-only inputs, mock X-ray
lightcones, and the limitations of the model (it smooths compact structures, and
box-integrated luminosities are biased low).

## What is in this repository

| Path | Contents |
|---|---|
| `paper_experiments/` | The scripts that produced every result, table and figure of the paper: preprocessing, training, tiled inference, metrics, baselines and all analyses. Start with its [README](paper_experiments/README.md). |
| `cosmomap/` | An installable package with the model components those scripts use (the FiLM layer, the residual blocks and the U-Net), the same normalisation and losses, a small training and evaluation pipeline, and X-ray utilities. |
| `configs/` | `default.yaml` (CosmoMap), `sigma.yaml` (CosmoMap-σ) and `synthetic.yaml` (a tiny demo). |
| `tests/` | Unit and end-to-end tests of the package and of the building blocks of the paper scripts. |
| `docs/METHOD.md` | A short guide from each part of the method to the code that implements it. |

The numbers in the paper come from `paper_experiments/`. The package pipeline in
`cosmomap/` uses the same network, normalisation, loss and optimiser settings, but it is a
compact reference implementation: it loads data into memory and has not been used to
reproduce the paper's numbers.

## Installation

```bash
git clone https://github.com/amanbh1/CosmoMap.git
cd CosmoMap
pip install -e ".[dev]"          # add ,paper for the extra dependencies of paper_experiments/
```

Python 3.10 or later and TensorFlow 2.16 or later (Keras 3). The paper's runs used
Python 3.12 and TensorFlow 2.21 on one NVIDIA RTX 4070 (12 GB).

## Quickstart without data

The package can generate small synthetic boxes, so the whole pipeline runs in a minute
on a CPU. The synthetic fields only exercise the code; they are not a physical model.

```bash
# train a tiny model
cosmomap-train --config configs/synthetic.yaml --synthetic

# accuracy overall and by density regime, and interval coverage
cosmomap-evaluate --config outputs/synthetic/config.yaml \
                  --model outputs/synthetic/final_model.keras --synthetic

# a simple mock X-ray surface-brightness map (FITS)
python scripts/make_lightcone.py --config outputs/synthetic/config.yaml \
                                 --model outputs/synthetic/final_model.keras --synthetic
```

Run the tests with

```bash
pytest
```

## Reproducing the paper

The paper uses the 3D grids of the
[CAMELS Multifield Dataset](https://camels-multifield-dataset.readthedocs.io/) for the
IllustrisTNG LH set at 128³ and z = 0, 0.5, 1, 1.5 and 2 (`Mcdm`, `Vcdm`, `T`, `ne`;
about 34 GB per redshift). [`paper_experiments/README.md`](paper_experiments/README.md)
lists the files, the order of the scripts and the command for each model. In short:

```bash
cd paper_experiments
bash download.sh                 # CAMELS grids -> ~/camels
python prep.py                   # normalise, fixed 800/100/100 split
python train.py --name mse_z0_s0 --seed 0 --zs 0.0 --loss mse     # about 3 h on an RTX 4070
python predict.py --name mse_z0_s0 --mc 0
python metrics.py mse_z0_s0
```

Training on a GPU is not bit-reproducible, so retrained models differ slightly from the
paper's; the seed-to-seed scatter is given in the paper (below 0.5 per cent in R² and RMSE
for CosmoMap).

## Training with the package on CAMELS data

```bash
cosmomap-train --config configs/default.yaml      # CosmoMap (MSE loss)
cosmomap-train --config configs/sigma.yaml        # CosmoMap-σ (likelihood loss, uncertainties)
```

Set `data.data_root` to the folder with the CAMELS grids. The loader reads the simulations
into memory, so `data.max_sims` (100 in the shipped configs) controls how many are used;
for the full set use the memory-mapped scripts in `paper_experiments/`.

## Citation

If you use this code, please cite the paper:

> A. Bhutani and M. Lieu, *CosmoMap: Predicting Hot Gas Temperature and Density from Dark
> Matter with a Conditioned 3D U-Net*.

and the CAMELS project and Multifield Dataset, whose simulations it is trained on
(Villaescusa-Navarro et al. 2021, 2022).

## Acknowledgements

We thank the CAMELS collaboration for making their simulations publicly available.

## License

[MIT](LICENSE) © 2025 Aman Bhutani
