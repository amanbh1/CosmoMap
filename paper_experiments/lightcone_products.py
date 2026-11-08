"""Write the example lightcone as survey-style products:
FITS images with a TAN world coordinate system (noise-free surface brightness and Poisson counts,
truth and prediction) and FITS source catalogues of detected sources.

    python lightcone_products.py   ->  ~/camels/runs/lightcone_products/
"""
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from common import RUNS

NPIX, FOV = 256, 1.0
out = RUNS / "lightcone_products"
out.mkdir(exist_ok=True)
d = np.load(RUNS / "lightcone_example.npz")

w = WCS(naxis=2)
w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
w.wcs.crval = [150.0, 2.0]            # arbitrary field centre
w.wcs.crpix = [NPIX / 2 + 0.5] * 2
w.wcs.cdelt = [-FOV / NPIX, FOV / NPIX]
w.wcs.cunit = ["deg", "deg"]
hdr = w.to_header()
hdr["BUNIT"] = "arbitrary"
hdr["COMMENT"] = "CosmoMap mock X-ray lightcone, 0.05<z<2.25, 30 arcsec FWHM PSF"

for key in ("truth", "pred"):
    fits.PrimaryHDU(d[key].T.astype(np.float32), header=hdr).writeto(out / f"sb_{key}.fits", overwrite=True)
    h2 = hdr.copy(); h2["BUNIT"] = "counts"
    fits.PrimaryHDU(d[f"counts_{key}"].T.astype(np.float32), header=h2).writeto(out / f"counts_{key}.fits", overwrite=True)
    cen = d["cen_t" if key == "truth" else "cen_p"]
    if len(cen):
        sky = w.pixel_to_world(cen[:, 0], cen[:, 1])
        Table({"x_pix": cen[:, 0], "y_pix": cen[:, 1], "ra_deg": sky.ra.deg, "dec_deg": sky.dec.deg}).write(
            out / f"catalogue_{key}.fits", overwrite=True)
print("wrote", sorted(p.name for p in out.iterdir()))
