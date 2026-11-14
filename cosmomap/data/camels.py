"""Loader for the 3D grids of the CAMELS Multifield Dataset.

Expected files under ``config.data.data_root`` (the names of the public release)::

    Grids_<field>_IllustrisTNG_LH_128_z=<z>.npy   # (n_sims, R, R, R) float32, one per field and redshift
    params_LH_IllustrisTNG.txt                    # one row per simulation:
                                                  # Omega_m sigma_8 A_SN1 A_AGN1 A_SN2 A_AGN2

Shorter names (``<field>_z=<z>.npy``) and a parameter table with a header row are
also accepted. See the README for where to download the data.

This loader reads the requested simulations into memory, which suits experiments
on a subset (``data.max_sims``). The full Latin-hypercube set at five redshifts is
about 170 GB as float32: the scripts in ``paper_experiments/`` handle it with
memory-mapped, pre-normalised float16 arrays instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cosmomap.config import Config
from cosmomap.data.container import CamelsData
from cosmomap.data.normalization import standardize_columns

_FIELD_PATTERNS = (
    "Grids_{field}_IllustrisTNG_LH_{res}_z={z}.npy",
    "{field}_z={z}.npy",
    "samples_3d_{field}_z={z}.npy",
    "{field}_z{z}.npy",
)
_PARAM_PATTERNS = ("params_LH_IllustrisTNG.txt", "CosmoAstroSeed*.txt", "*atin*ypercube*.txt", "*params*.txt")
#: Column order of a parameter table without a header row (CAMELS convention).
_PARAM_ORDER = ("Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2")


def _find_param_file(root: Path) -> Path:
    for pattern in _PARAM_PATTERNS:
        hits = sorted(root.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No CAMELS parameter table found under {root} (looked for {', '.join(_PARAM_PATTERNS)})."
    )


def _find_field_file(root: Path, field: str, z: float, resolution: int) -> Path:
    for pattern in _FIELD_PATTERNS:
        for z_str in (f"{z:.1f}", f"{z:g}", str(z)):
            candidate = root / pattern.format(field=field, z=z_str, res=resolution)
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"No file for field '{field}' at z={z} under {root} "
        f"(expected e.g. {_FIELD_PATTERNS[0].format(field=field, z=f'{z:.1f}', res=resolution)})."
    )


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _read_params(path: Path, names: list[str]) -> np.ndarray:
    """Return an ``(n_sims, len(names))`` array of the requested parameters."""
    first = path.read_text().split("\n", 1)[0].replace("#", " ").split()
    if not all(_is_number(tok) for tok in first):  # header row with column names
        table = np.genfromtxt(path, names=True, dtype=None, encoding=None)
        available = getattr(table.dtype, "names", None) or ()
        missing = [p for p in names if p not in available]
        if missing:
            raise KeyError(f"parameter table {path} has no column(s) {missing}; available: {available}")
        return np.stack([np.asarray(table[p], dtype=np.float32) for p in names], axis=1)
    table = np.atleast_2d(np.loadtxt(path, dtype=np.float32))
    unknown = [p for p in names if p not in _PARAM_ORDER]
    if unknown:
        raise KeyError(f"{path} has no header and {unknown} are not standard CAMELS parameters {_PARAM_ORDER}")
    return table[:, [_PARAM_ORDER.index(p) for p in names]]


def load_camels(config: Config, max_sims: int | None = None) -> CamelsData:
    """Load CAMELS fields into a :class:`CamelsData` container.

    Parameters
    ----------
    config:
        Pipeline configuration (uses ``data_root``, ``fields_*``, ``redshifts``,
        ``cosmo_params``, ``resolution``, ``max_sims`` and ``include_redshift_in_cond``).
    max_sims:
        If given, overrides ``config.data.max_sims``: only the first ``max_sims``
        simulations are loaded.
    """
    root = Path(config.data.data_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(
            f"data_root '{root}' does not exist. Provide CAMELS data or use the "
            f"synthetic generator (--synthetic)."
        )
    max_sims = config.data.max_sims if max_sims is None else max_sims
    params = _read_params(_find_param_file(root), list(config.data.cosmo_params))
    n_sims = params.shape[0] if max_sims is None else min(max_sims, params.shape[0])
    params = params[:n_sims]

    zs = config.data.redshifts
    fields_all = config.data.fields_in + config.data.fields_out
    loaded = {
        (f, z): np.load(_find_field_file(root, f, z, config.data.resolution), mmap_mode="r")
        for f in fields_all
        for z in zs
    }

    # Sample order: simulation outer, redshift inner (sample = sim * n_z + z index),
    # the same order as the synthetic generator.
    buffers: dict[str, list[np.ndarray]] = {f: [] for f in fields_all}
    cond_rows: list[np.ndarray] = []
    sim_id_rows: list[int] = []
    cond_names = list(config.data.cosmo_params) + (
        ["z"] if config.data.include_redshift_in_cond else []
    )
    for s in range(n_sims):
        for z in zs:
            for f in fields_all:
                buffers[f].append(np.asarray(loaded[(f, z)][s], dtype=np.float32))
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
