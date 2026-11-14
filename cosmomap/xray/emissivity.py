"""Thermal-bremsstrahlung X-ray emissivity and surface-brightness projection.

The dominant X-ray emission from the hot intracluster medium is thermal
bremsstrahlung, for which the bolometric emissivity scales as::

    eps_X = A * n_e^2 * sqrt(T)

with ``A = 1.42e-27`` (cgs), a Gaunt factor of unity and the ion term replaced by
``n_e``. Computed from voxel-averaged fields it neglects sub-voxel clumping, so it
is a proxy for comparing predicted and true fields, not an absolute luminosity.
"""

from __future__ import annotations

import numpy as np

#: Bremsstrahlung normalisation constant (erg s^-1 cm^3).
EMISSIVITY_CONST = 1.42e-27


def compute_emissivity(ne: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """X-ray emissivity ``eps_X = A * n_e^2 * sqrt(T)`` (element-wise)."""
    ne = np.asarray(ne, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    return EMISSIVITY_CONST * np.square(ne) * np.sqrt(np.maximum(temperature, 0.0))


def surface_brightness(
    emissivity: np.ndarray,
    axis: int = -1,
    redshift: float = 0.0,
    voxel_depth: float = 1.0,
) -> np.ndarray:
    """Project an emissivity volume to a surface-brightness map.

    Integrates along the line-of-sight ``axis`` and applies the cosmological
    ``(1 + z)^-4`` surface-brightness dimming.
    """
    emissivity = np.asarray(emissivity, dtype=np.float64)
    integrated = np.sum(emissivity, axis=axis) * voxel_depth
    return integrated / (1.0 + redshift) ** 4
