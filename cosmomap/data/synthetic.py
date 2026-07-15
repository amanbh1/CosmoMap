"""Synthetic, physically-flavoured CAMELS-like data.

This lets the whole pipeline run end-to-end (train, evaluate, lightcone) with no
download, so reviewers can ``pip install -e . && pytest`` or train a toy model in
seconds. Fields are *correlated* the way the real ones are:

* dark-matter density is a lognormal field built from a Gaussian random field;
* velocity magnitude tracks the density gradient;
* electron density follows the gas (~ DM) density with a small bias and scatter;
* temperature follows a shock-heating relation ``T ~ rho^(2/3)`` modulated by the
  AGN-feedback parameter,

so the network has genuine structure to learn and the physics-informed losses
(positive ``n_e``, density-temperature correlation) are meaningful.
"""

from __future__ import annotations

import numpy as np

from cosmomap.config import Config
from cosmomap.data.container import CamelsData
from cosmomap.data.normalization import standardize_columns

#: Approximate CAMELS Latin-hypercube parameter ranges (for realistic conditioning).
_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "Omega_m": (0.1, 0.5),
    "sigma_8": (0.6, 1.0),
    "A_SN1": (0.25, 4.0),
    "A_AGN1": (0.25, 4.0),
    "A_SN2": (0.5, 2.0),
    "A_AGN2": (0.5, 2.0),
}


def _gaussian_random_field(r: int, rng: np.random.Generator, n_index: float = -3.0) -> np.ndarray:
    """Zero-mean, unit-variance GRF with a power-law spectrum ``P(k) ~ k^n``."""
    white = rng.standard_normal((r, r, r)).astype(np.float32)
    fk = np.fft.fftn(white)
    k = np.fft.fftfreq(r) * r
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    kk = np.sqrt(kx**2 + ky**2 + kz**2)
    kk[0, 0, 0] = 1.0
    amp = kk ** (n_index / 2.0)
    amp[0, 0, 0] = 0.0  # remove the DC mode
    field = np.fft.ifftn(fk * amp).real.astype(np.float32)
    field -= field.mean()
    std = field.std()
    if std > 0:
        field /= std
    return field.astype(np.float32)


def make_synthetic_camels(config: Config, n_sims: int = 8, seed: int = 0) -> CamelsData:
    """Generate ``n_sims`` simulations x ``len(redshifts)`` snapshots of fake data."""
    rng = np.random.default_rng(seed)
    r = config.data.resolution
    zs = config.data.redshifts
    pnames = list(config.data.cosmo_params)

    params = np.stack(
        [rng.uniform(*_PARAM_RANGES.get(p, (0.0, 1.0)), size=n_sims) for p in pnames],
        axis=1,
    ).astype(np.float32)

    fields_all = config.data.fields_in + config.data.fields_out
    buffers: dict[str, list[np.ndarray]] = {f: [] for f in fields_all}
    cond_rows: list[np.ndarray] = []
    sim_id_rows: list[int] = []
    cond_names = pnames + (["z"] if config.data.include_redshift_in_cond else [])

    def _p(sim: int, name: str, default: float) -> float:
        return float(params[sim, pnames.index(name)]) if name in pnames else default

    for s in range(n_sims):
        omega_m = _p(s, "Omega_m", 0.3)
        a_agn = _p(s, "A_AGN1", 1.0)
        for z in zs:
            delta = _gaussian_random_field(r, rng, n_index=-3.0)
            growth = 1.0 / (1.0 + z)  # structure grows toward low redshift
            log_rho = (0.6 + 0.5 * omega_m) * growth * delta

            mcdm = np.power(10.0, log_rho).astype(np.float32)
            vcdm = np.abs(
                _gaussian_random_field(r, rng, n_index=-5.0) + 0.3 * delta
            ).astype(np.float32)
            ne = np.power(
                10.0, 0.9 * log_rho - 3.0 + 0.15 * rng.standard_normal((r, r, r))
            ).astype(np.float32)
            temp = np.power(
                10.0,
                6.0 + 0.66 * log_rho + 0.1 * (a_agn - 1.0) + 0.1 * rng.standard_normal((r, r, r)),
            ).astype(np.float32)

            sample = {"Mcdm": mcdm, "Vcdm": vcdm, "ne": ne, "T": temp}
            for f in fields_all:
                if f not in sample:
                    raise KeyError(
                        f"synthetic generator does not know field '{f}'; "
                        f"supported: {sorted(sample)}"
                    )
                buffers[f].append(sample[f])

            row = list(params[s])
            if config.data.include_redshift_in_cond:
                row.append(float(z))
            cond_rows.append(np.asarray(row, dtype=np.float32))
            sim_id_rows.append(s)

    fields = {f: np.stack(v, axis=0).astype(np.float32) for f, v in buffers.items()}
    cond = standardize_columns(np.stack(cond_rows, axis=0))
    return CamelsData(
        fields=fields,
        cond=cond,
        resolution=r,
        redshifts=list(zs),
        n_sims=n_sims,
        cond_names=cond_names,
        sim_ids=np.asarray(sim_id_rows, dtype=np.int32),
    )
