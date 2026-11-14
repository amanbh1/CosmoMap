"""Mock X-ray lightcones from true vs predicted fields, multi-redshift model.

For each of N test simulations we build a 1 deg x 1 deg lightcone from z=0.05 to z~2.25:
  * the line of sight is cut into 25 Mpc/h slabs at comoving distance chi; each slab uses the
    snapshot (z = 0, 0.5, 1, 1.5, 2) nearest in redshift, with a random cube symmetry and periodic
    shift (identical for truth and prediction);
  * each slab's emissivity is projected along the line of sight and sampled at transverse comoving
    positions chi * theta (bilinear, periodic), giving surface brightness
    S = (1+z)^-4 / (4 pi) * sum eps * dl_proper;
  * ne is treated as comoving and converted to physical with (1+z)^3; only RELATIVE fluxes matter
    because truth and prediction go through the identical pipeline;
  * the noise-free maps are convolved with a 30" FWHM Gaussian PSF, scaled to counts so the
    median truth map has a fixed signal-to-background, a uniform background is added and the
    same Poisson realisation is drawn (common random numbers) for truth and prediction;
  * sources are detected as connected regions more than 5 sigma above a local (median-filtered)
    background after PSF smoothing, and matched between maps within 2 PSF FWHM.

    python lightcone.py [n_lightcones]      (the paper uses 10)
Writes RUNS/lightcone.json (per-lightcone and pooled statistics, with per-source match flags) and
RUNS/lightcone_example.npz (maps, counts and detections of the first lightcone).
"""
import json
import sys

import numpy as np
from scipy import ndimage
from scipy.integrate import quad

from common import MAIN_ALLZ, RUNS, load_params, load_stats, prep_path, split, to_log_physical

RUN = MAIN_ALLZ
SNAPS = [0.0, 0.5, 1.0, 1.5, 2.0]
BOX, RES = 25.0, 128
NPIX, FOV_DEG = 256, 1.0
PSF_FWHM_ARCSEC = 30.0
ZMIN, ZMAX = 0.05, 2.25
C_KMS = 299792.458


def chi_of_z(z, om):
    return quad(lambda x: C_KMS / 100.0 / np.sqrt(om * (1 + x) ** 3 + 1 - om), 0, z)[0]  # Mpc/h


def z_of_chi_table(om):
    zs = np.linspace(0, 3, 600)
    return zs, np.array([chi_of_z(z, om) for z in zs])


def sym(a, rng):
    perm = rng.permutation(3)
    a = np.transpose(a, perm)
    for ax in range(3):
        if rng.integers(2):
            a = np.flip(a, ax)
    return np.roll(a, tuple(rng.integers(0, RES, 3)), axis=(0, 1, 2))


def sample_periodic(img, x, y):
    """Bilinear periodic sampling of a 128x128 map at comoving positions x, y (Mpc/h)."""
    gx, gy = x / BOX * RES, y / BOX * RES
    i0, j0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = gx - i0, gy - j0
    i0 %= RES; j0 %= RES
    i1, j1 = (i0 + 1) % RES, (j0 + 1) % RES
    return (img[i0, j0] * (1 - fx) * (1 - fy) + img[i1, j0] * fx * (1 - fy)
            + img[i0, j1] * (1 - fx) * fy + img[i1, j1] * fx * fy)


def detect(counts, psf_sigma_pix, bkg_box=33):
    """Sliding-cell detection: PSF-smoothed counts against a local background (median filter over
    bkg_box pixels, about 2.2 arcmin), threshold at 5 sigma of the smoothed Poisson noise."""
    sm = ndimage.gaussian_filter(counts, psf_sigma_pix)
    local = ndimage.median_filter(sm, size=bkg_box, mode="wrap")
    sig = np.sqrt(local / (4 * np.pi * psf_sigma_pix ** 2))
    lab, n = ndimage.label(sm > local + 5 * sig)
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0)
    cen = np.array(ndimage.center_of_mass(sm, lab, range(1, n + 1)))
    flux = ndimage.sum(counts - local, lab, range(1, n + 1))
    return cen, flux


def main():
    nlc = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    st = load_stats()
    P = load_params()
    test = split()["test"]
    tru = {(f, z): np.load(prep_path(f, z), mmap_mode="r") for f in ("T", "ne") for z in SNAPS}
    pre = {(f, z): np.load(RUNS / RUN / "pred" / f"mu_{f}_z{z:.1f}.npy", mmap_mode="r") for f in ("T", "ne") for z in SNAPS}
    theta = (np.arange(NPIX) + 0.5) / NPIX * np.deg2rad(FOV_DEG)
    TX, TY = np.meshgrid(theta, theta, indexing="ij")
    pix_arcsec = FOV_DEG * 3600 / NPIX
    psf_sigma = PSF_FWHM_ARCSEC / 2.355 / pix_arcsec
    out = {"lightcones": []}
    rng_noise = np.random.default_rng(123)

    for n in range(nlc):
        k = n  # n-th test simulation
        sim = test[k]
        om = float(P[sim, 0])
        ztab, chitab = z_of_chi_table(om)
        chi0, chi1 = np.interp(ZMIN, ztab, chitab), np.interp(ZMAX, ztab, chitab)
        edges = np.arange(chi0, chi1, BOX)
        rng = np.random.default_rng(1000 + n)
        maps = {"truth": np.zeros((NPIX, NPIX)), "pred": np.zeros((NPIX, NPIX))}
        cache = {}
        for c_lo in edges:
            c_mid = c_lo + BOX / 2
            z = float(np.interp(c_mid, chitab, ztab))
            zs = min(SNAPS, key=lambda s: abs(s - z))
            if zs not in cache:
                lT = to_log_physical(np.asarray(tru[("T", zs)][sim], np.float32), "T", st)
                lne = to_log_physical(np.asarray(tru[("ne", zs)][sim], np.float32), "ne", st)
                pT = np.asarray(pre[("T", zs)][k], np.float32)
                pne = np.asarray(pre[("ne", zs)][k], np.float32)
                cache[zs] = (lT, lne, pT, pne)
            lT, lne, pT, pne = cache[zs]
            seed = int(rng.integers(1 << 30))
            for key, (a, b) in (("truth", (lT, lne)), ("pred", (pT, pne))):
                # physical emissivity (arbitrary constant), ne comoving -> physical
                leps = 2 * (b + 3 * np.log10(1 + zs)) + 0.5 * a
                eps = sym(10.0 ** leps.astype(np.float64), np.random.default_rng(seed))
                proj = eps.sum(2) * (BOX / RES) / (1 + z)  # proper path length per voxel
                sb = sample_periodic(proj, c_mid * TX, c_mid * TY) / (4 * np.pi * (1 + z) ** 4)
                maps[key] += sb
        # PSF, counts, background, Poisson (common random numbers)
        conv = {kk: ndimage.gaussian_filter(v, psf_sigma) for kk, v in maps.items()}
        scale = 2.0 / np.median(conv["truth"])      # median truth pixel -> 2 source counts (faint diffuse emission)
        bkg = 40.0                                   # background counts per pixel
        u = rng_noise.random((NPIX, NPIX))
        from scipy.stats import poisson
        counts = {kk: poisson.ppf(u, v * scale + bkg) for kk, v in conv.items()}
        cen_t, fl_t = detect(counts["truth"], psf_sigma)
        cen_p, fl_p = detect(counts["pred"], psf_sigma)
        match_r = 2 * PSF_FWHM_ARCSEC / pix_arcsec
        matched, ratios = [], []
        for i, c in enumerate(cen_t):
            if len(cen_p):
                d = np.hypot(*(cen_p - c).T)
                j = int(np.argmin(d))
                if d[j] < match_r:
                    matched.append(i)
                    ratios.append(float(np.log10(max(fl_p[j], 1) / max(fl_t[i], 1))))
        purity_hits, real = 0, []
        for c in cen_p:
            hit = bool(len(cen_t) and np.min(np.hypot(*(cen_t - c).T)) < match_r)
            real.append(hit)
            purity_hits += hit
        lt, lp = np.log10(conv["truth"]), np.log10(conv["pred"])
        rec = dict(sim=int(sim), n_slabs=len(edges), n_true=len(cen_t), n_pred=len(cen_p),
                   completeness=len(matched) / max(len(cen_t), 1), purity=purity_hits / max(len(cen_p), 1),
                   flux_logratio=ratios, flux_true=fl_t.tolist(), flux_pred=fl_p.tolist(), matched_true=matched, pred_real=real,
                   map_pearson_log=float(np.corrcoef(lt.ravel(), lp.ravel())[0, 1]),
                   total_flux_logratio=float(np.log10(maps["pred"].sum() / maps["truth"].sum())))
        out["lightcones"].append(rec)
        if n == 0:  # example lightcone for the figure and the FITS products
            np.savez_compressed(RUNS / "lightcone_example.npz", truth=conv["truth"], pred=conv["pred"],
                                counts_truth=counts["truth"], counts_pred=counts["pred"], cen_t=cen_t, cen_p=cen_p)
        print(f"lc {n}: slabs {len(edges)} n_true {len(cen_t)} n_pred {len(cen_p)} compl {rec['completeness']:.2f} "
              f"purity {rec['purity']:.2f} map r {rec['map_pearson_log']:.3f} total {rec['total_flux_logratio']:+.2f} "
              f"med flux ratio {np.median(ratios) if ratios else float('nan'):+.2f}", flush=True)
    L = out["lightcones"]
    allr = np.concatenate([r["flux_logratio"] for r in L])
    out["summary"] = dict(completeness=float(np.mean([r["completeness"] for r in L])),
                          purity=float(np.mean([r["purity"] for r in L])),
                          n_true=int(np.sum([r["n_true"] for r in L])), n_pred=int(np.sum([r["n_pred"] for r in L])),
                          flux_logratio_median=float(np.median(allr)), flux_logratio_sd=float(np.std(allr)),
                          map_pearson_log=float(np.mean([r["map_pearson_log"] for r in L])),
                          total_flux_logratio=float(np.mean([r["total_flux_logratio"] for r in L])))
    print("SUMMARY", out["summary"])
    (RUNS / "lightcone.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
