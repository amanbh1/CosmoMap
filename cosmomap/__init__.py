"""CosmoMap: field-level deep learning for baryon prediction on CAMELS.

A 3D FiLM-conditioned U-Net that maps dark-matter fields (density and velocity
magnitude) to gas temperature and electron number density, conditioned on the
six CAMELS cosmological and feedback parameters and the redshift, with optional
per-voxel uncertainties and X-ray emissivity utilities.

The results of the paper (Bhutani & Lieu) are produced by the scripts in
``paper_experiments/``, which use the model components of this package; see the
README and ``docs/METHOD.md``.
"""

from cosmomap.config import Config

__version__ = "0.1.0"
__all__ = ["Config", "__version__"]
