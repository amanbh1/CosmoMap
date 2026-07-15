"""In-memory container shared by the synthetic generator and the real loader."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CamelsData:
    """A set of 3D simulation sub-boxes plus their conditioning vectors.

    Attributes
    ----------
    fields:
        Mapping ``field_name -> array`` of shape ``(N, R, R, R)`` (float32),
        where ``N`` is the number of samples (simulations x redshifts) and
        ``R`` is :attr:`resolution`. Fields are stored in *physical* units;
        normalisation is applied per patch at sampling time.
    cond:
        Array ``(N, n_cond)`` of already-normalised conditioning vectors
        (the CAMELS parameters, optionally with redshift appended).
    resolution:
        Voxels per side of each stored sub-box.
    redshifts:
        Redshift snapshots represented in the sample axis.
    n_sims:
        Number of distinct simulations (``N == n_sims * len(redshifts)``).
    cond_names:
        Human-readable names for the columns of :attr:`cond`.
    """

    fields: dict[str, np.ndarray]
    cond: np.ndarray
    resolution: int
    redshifts: list[float]
    n_sims: int
    cond_names: list[str]
    #: Simulation index for each sample (length N), enabling simulation-level
    #: splitting so a simulation's redshifts never straddle train/val/test.
    sim_ids: np.ndarray

    def __post_init__(self) -> None:
        n = self.cond.shape[0]
        if self.sim_ids.shape[0] != n:
            raise ValueError(f"sim_ids has {self.sim_ids.shape[0]} entries but cond has {n}")
        for name, arr in self.fields.items():
            if arr.shape[0] != n:
                raise ValueError(
                    f"field '{name}' has {arr.shape[0]} samples but cond has {n}"
                )
            if arr.ndim != 4 or arr.shape[1] != arr.shape[2] != arr.shape[3]:
                raise ValueError(f"field '{name}' must be (N, R, R, R); got {arr.shape}")

    @property
    def n_samples(self) -> int:
        return self.cond.shape[0]

    @property
    def n_cond(self) -> int:
        return self.cond.shape[1]
