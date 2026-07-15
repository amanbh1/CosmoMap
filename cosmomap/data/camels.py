"""Loader for real CAMELS 3D fields.

Expected layout under ``config.data.data_root``:

    <root>/
      CosmoAstroSeed_*.txt                 # whitespace table; one row per simulation,
                                           # with columns Omega_m sigma_8 A_SN1 A_AGN1 A_SN2 A_AGN2
      <field>_z=<z>.npy                    # array of shape (n_sims, R, R, R), float32
                                           # e.g. Mcdm_z=0.0.npy, ne_z=1.0.npy, ...

See ``scripts/download_camels.md`` for how to obtain and arrange the data. The
full Latin-hypercube set is large; use ``max_sims`` to load a subset for
development on a workstation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cosmomap.config import Config
from cosmomap.data.container import CamelsData
from cosmomap.data.normalization import standardize_columns

_FIELD_PATTERNS = ("{field}_z={z}.npy", "samples_3d_{field}_z={z}.npy", "{field}_z{z}.npy")
_PARAM_PATTERNS = ("CosmoAstroSeed*.txt", "*atin*ypercube*.txt", "*params*.txt")


def _find_param_file(root: Path) -> Path:
    for pattern in _PARAM_PATTERNS:
        hits = sorted(root.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No CAMELS parameter table found under {root} "
        f"(looked for {', '.join(_PARAM_PATTERNS)}). See scripts/download_camels.md."
    )


def _find_field_file(root: Path, field: str, z: float) -> Path:
    for pattern in _FIELD_PATTERNS:
        for z_str in (f"{z:.1f}", f"{z:g}", str(z)):
            candidate = root / pattern.format(field=field, z=z_str)
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"No file for field '{field}' at z={z} under {root}. See scripts/download_camels.md."
    )


def load_camels(config: Config, max_sims: int | None = None) -> CamelsData:
    """Load real CAMELS fields into a :class:`CamelsData` container.

    Parameters
    ----------
    config:
        Pipeline configuration (uses ``data_root``, ``fields_*``, ``redshifts``,
        ``cosmo_params``, ``resolution`` and ``include_redshift_in_cond``).
    max_sims:
        If given, only the first ``max_sims`` simulations are loaded (handy for
        development; the full set needs a lot of RAM).
    """
    root = Path(config.data.data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"data_root '{root}' does not exist. Provide CAMELS data or use the "
            f"synthetic generator (--synthetic). See scripts/download_camels.md."
        )

    table = np.genfromtxt(_find_param_file(root), names=True, dtype=None, encoding=None)
    try:
        params = np.stack(
            [np.asarray(table[p], dtype=np.float32) for p in config.data.cosmo_params], axis=1
        )
    except (ValueError, KeyError) as exc:  # missing column
        available = getattr(table.dtype, "names", None)
        raise KeyError(
            f"parameter table is missing one of {config.data.cosmo_params}; "
            f"available columns: {available}"
        ) from exc

    n_sims = params.shape[0] if max_sims is None else min(max_sims, params.shape[0])
    params = params[:n_sims]

    zs = config.data.redshifts
    fields_all = config.data.fields_in + config.data.fields_out

    buffers: dict[str, list[np.ndarray]] = {f: [] for f in fields_all}
    cond_rows: list[np.ndarray] = []
    sim_id_rows: list[int] = []
    cond_names = list(config.data.cosmo_params) + (
        ["z"] if config.data.include_redshift_in_cond else []
    )

    for z in zs:
        loaded = {f: np.load(_find_field_file(root, f, z), mmap_mode="r") for f in fields_all}
        for s in range(n_sims):
            for f in fields_all:
                buffers[f].append(np.asarray(loaded[f][s], dtype=np.float32))
            row = list(params[s])
            if config.data.include_redshift_in_cond:
                row.append(float(z))
            cond_rows.append(np.asarray(row, dtype=np.float32))
            sim_id_rows.append(s)

    fields = {f: np.stack(v, axis=0) for f, v in buffers.items()}
    resolution = next(iter(fields.values())).shape[1]
    cond = standardize_columns(np.stack(cond_rows, axis=0))
    return CamelsData(
        fields=fields,
        cond=cond,
        resolution=resolution,
        redshifts=list(zs),
        n_sims=n_sims,
        cond_names=cond_names,
        sim_ids=np.asarray(sim_id_rows, dtype=np.int32),
    )
