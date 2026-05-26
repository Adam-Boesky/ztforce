"""
Synthetic benchmark comparing PSF approaches for ZTF forced photometry.

Simulates realistic ZTF science image conditions and measures flux-recovery
accuracy for four candidate approaches, to inform the choice of PSF model
used in ztforce.

Conditions simulated
--------------------
- Plate scale: 1.01 arcsec/px
- Moffat PSF (beta=3.5), FWHM varied: 2.0, 3.0, 4.0 px
- Realistic sky background (200 ADU/px) + Gaussian readout noise (10 e-)
- 120 sources spanning SNR = 3–300

Approaches compared
-------------------
1. Aperture photometry  — circular aperture + aperture correction from PSF model
2. Analytic Moffat fit  — fit Moffat PSF with position fixed, only flux free
3. PSF stamp (perfect)  — pre-built stamp matching truth PSF exactly
4. PSF stamp (+0.3px)   — stamp with 0.3px FWHM error (realistic spatial variation)
5. ePSF                 — empirical PSF built from 40 bright stars, forced on rest

Results summary (FWHM=2px, SNR>30)
-----------------------------------
Aperture          ~3%   bias
Moffat            ~1%   bias
PSF stamp perfect ~0.5% bias
PSF stamp +0.3px  ~7%   bias  (shows why spatial variation matters)
ePSF              ~89%  bias  (EPSFBuilder normalization issues at low oversampling)

Conclusion
----------
The ZTF sciimgdao.psf sidecar provides a spatially-varying PSF at every
image position with no extra star-finding cost. Combined with a matched-filter
estimator (ztforce's approach), this avoids the mismatch bias of a
center-only stamp while being far simpler than building an ePSF.

Usage
-----
    conda run -n long_transients python scripts/psf_approach_benchmark.py
"""

import time
import warnings

import matplotlib

matplotlib.use("Agg")
import numpy as np
import sep
from astropy.modeling.models import Moffat2D
from astropy.nddata import NDData
from astropy.table import Table
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.psf import EPSFBuilder, EPSFFitter, EPSFStars, PSFPhotometry, extract_stars

warnings.filterwarnings("ignore")

RNG = np.random.default_rng(42)

# ── ZTF realistic parameters ──────────────────────────────────────────────────
GAIN = 6.2
READNOISE = 10.0
SKY_ADU = 200.0
MAGZP = 26.3
IMAGE_SIZE = 512
MOFFAT_BETA = 3.5
N_STARS_FOR_EPSF = 40
N_TARGET_STARS = 80


def make_moffat_psf_image(fwhm_px, size=51, beta=MOFFAT_BETA):
    """Build a normalized Moffat PSF stamp."""
    cy, cx = size // 2, size // 2
    y, x = np.mgrid[0:size, 0:size]
    gamma = fwhm_px / (2 * np.sqrt(2 ** (1 / beta) - 1))
    psf = Moffat2D(amplitude=1.0, x_0=cx, y_0=cy, gamma=gamma, alpha=beta)(x, y)
    return psf / psf.sum()


def make_synthetic_image(fwhm_px, n_stars=N_STARS_FOR_EPSF + N_TARGET_STARS):
    """Create a realistic synthetic ZTF-like science image."""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    psf_size = 51
    psf_stamp = make_moffat_psf_image(fwhm_px, size=psf_size)

    margin = psf_size // 2 + 5
    xs = RNG.integers(margin, IMAGE_SIZE - margin, size=n_stars)
    ys = RNG.integers(margin, IMAGE_SIZE - margin, size=n_stars)

    sky_noise = np.sqrt(SKY_ADU / GAIN)
    psf_peak = psf_stamp.max()
    snr_targets = np.logspace(np.log10(3), np.log10(300), n_stars)
    fluxes = snr_targets * sky_noise / psf_peak

    half = psf_size // 2
    for xi, yi, flux in zip(xs, ys, fluxes, strict=False):
        img[yi - half : yi + half + 1, xi - half : xi + half + 1] += flux * psf_stamp

    sky = RNG.poisson(SKY_ADU, size=(IMAGE_SIZE, IMAGE_SIZE)).astype(float)
    shot = RNG.normal(0, READNOISE / GAIN, size=(IMAGE_SIZE, IMAGE_SIZE))
    img = img + sky + shot

    sources = Table({"x": xs, "y": ys, "flux_true": fluxes, "snr_true": snr_targets})
    return np.ascontiguousarray(img, dtype=np.float64), sources, psf_stamp


def sky_subtract(img):
    """SEP background subtraction."""
    bkg = sep.Background(np.ascontiguousarray(img, dtype=np.float64), bw=64, bh=64, fw=3, fh=3)
    return img - bkg.back(), bkg.globalrms


# ── Approach 1: Aperture photometry ───────────────────────────────────────────


def approach_aperture(img_sub, sources, fwhm_px, bkg_rms):
    """Circular aperture photometry + aperture correction from a few bright stars."""
    r_aper = 2.5 * fwhm_px
    r_in, r_out = 4.0 * fwhm_px, 6.0 * fwhm_px  # noqa: F841

    positions = list(zip(sources["x"].astype(float), sources["y"].astype(float), strict=False))
    aper = CircularAperture(positions, r=r_aper)

    t0 = time.perf_counter()
    phot = aperture_photometry(img_sub, aper)
    elapsed = time.perf_counter() - t0

    psf_stamp = make_moffat_psf_image(fwhm_px, size=51)
    psf_aper = CircularAperture([(25.0, 25.0)], r=r_aper)
    aper_frac = float(aperture_photometry(psf_stamp, psf_aper)["aperture_sum"][0])
    aper_corr = 1.0 / aper_frac

    flux_measured = phot["aperture_sum"] * aper_corr
    return np.array(flux_measured), elapsed * 1000


# ── Approach 2: Analytic Moffat fit ──────────────────────────────────────────


def approach_moffat(img_sub, sources, fwhm_px, bkg_rms):
    """Fit a Moffat PSF at each forced position. Position fixed, only flux free."""
    from photutils.psf import ImagePSF

    gamma = fwhm_px / (2 * np.sqrt(2 ** (1 / MOFFAT_BETA) - 1))
    psf_size = int(np.ceil(fwhm_px * 10 / 2) * 2 + 1)
    if psf_size < 11:
        psf_size = 11
    cy, cx = psf_size // 2, psf_size // 2
    yy, xx = np.mgrid[0:psf_size, 0:psf_size]
    psf_data = Moffat2D(amplitude=1.0, x_0=cx, y_0=cy, gamma=gamma, alpha=MOFFAT_BETA)(xx, yy)
    psf_data /= psf_data.sum()
    epsf_model = ImagePSF(psf_data)

    fit_shape = int(np.ceil(fwhm_px * 4 / 2) * 2 + 1)
    if fit_shape < 5:
        fit_shape = 5

    init_params = Table(
        {
            "x": sources["x"].astype(float),
            "y": sources["y"].astype(float),
            "flux_init": np.clip(sources["flux_true"] * 0.8, 1.0, None),
        }
    )

    t0 = time.perf_counter()
    phot = PSFPhotometry(epsf_model, fit_shape)
    result = phot(img_sub, init_params=init_params)
    elapsed = time.perf_counter() - t0

    return np.array(result["flux_fit"]), elapsed * 1000


# ── Approach 3: PSF stamp (pre-built file) ────────────────────────────────────


def approach_psfex_stamp(img_sub, sources, fwhm_px, bkg_rms, psf_error_fwhm=0.0):
    """
    Use a pre-built PSF stamp (as would come from sciimgdaopsfcent.fits).

    psf_error_fwhm adds a FWHM offset to simulate PSF mismatch due to spatial
    variation (realistic scenario when using a center-only stamp everywhere).
    Position fixed, only flux free.
    """
    from photutils.psf import ImagePSF

    psf_size = int(np.ceil(fwhm_px * 10 / 2) * 2 + 1)
    if psf_size < 11:
        psf_size = 11

    fwhm_used = fwhm_px + psf_error_fwhm
    psf_stamp = make_moffat_psf_image(fwhm_used, size=psf_size)
    epsf_model = ImagePSF(psf_stamp)

    fit_shape = int(np.ceil(fwhm_px * 4 / 2) * 2 + 1)
    if fit_shape < 5:
        fit_shape = 5

    init_params = Table(
        {
            "x": sources["x"].astype(float),
            "y": sources["y"].astype(float),
            "flux_init": np.clip(sources["flux_true"] * 0.8, 1.0, None),
        }
    )

    t0 = time.perf_counter()
    phot = PSFPhotometry(epsf_model, fit_shape)
    result = phot(img_sub, init_params=init_params)
    elapsed = time.perf_counter() - t0

    return np.array(result["flux_fit"]), elapsed * 1000


# ── Approach 4: ePSF built from image stars ───────────────────────────────────


def approach_epsf(img_sub, sources, fwhm_px, bkg_rms):
    """Build ePSF from bright isolated stars in the image, then forced photometry."""
    psf_star_sources = sources[:N_STARS_FOR_EPSF]
    target_sources = sources[N_STARS_FOR_EPSF:]

    psf_cutout_size = int(np.ceil(fwhm_px * 14 / 2) * 2 + 1)
    if psf_cutout_size < 15:
        psf_cutout_size = 15

    t0 = time.perf_counter()

    nd_data = NDData(img_sub)
    star_coords = Table(
        {
            "x": psf_star_sources["x"].astype(float),
            "y": psf_star_sources["y"].astype(float),
        }
    )
    stars = extract_stars(nd_data, star_coords, size=psf_cutout_size)
    stars = EPSFStars([s for s in stars if not np.any(~np.isfinite(s.data))])

    if len(stars) < 5:
        return np.full(len(target_sources), np.nan), 0.0

    fit_box = int(np.ceil(fwhm_px * 2 / 2) * 2 + 1)
    fitter = EPSFFitter(fit_boxsize=tuple([fit_box, fit_box]))
    builder = EPSFBuilder(fitter=fitter, oversampling=4, maxiters=10)
    epsf, _ = builder(stars)

    fit_shape = int(np.ceil(fwhm_px * 4 / 2) * 2 + 1)
    if fit_shape < 5:
        fit_shape = 5

    init_params = Table(
        {
            "x": target_sources["x"].astype(float),
            "y": target_sources["y"].astype(float),
            "flux_init": np.clip(target_sources["flux_true"] * 0.8, 1.0, None),
        }
    )

    phot = PSFPhotometry(epsf, fit_shape)
    result = phot(img_sub, init_params=init_params)
    elapsed = time.perf_counter() - t0

    return np.array(result["flux_fit"]), elapsed * 1000


# ── Run benchmark ─────────────────────────────────────────────────────────────


def _stats(frac_err, snr, snr_cut):
    mask = snr > snr_cut
    if not mask.any():
        return float("nan"), float("nan")
    fe = frac_err[mask]
    return float(np.nanmedian(np.abs(fe)) * 100), float(np.nanstd(fe) * 100)


def main():
    """Run the benchmark and print results."""
    print("=" * 70)
    print("ZTF Forced Photometry: PSF Approach Benchmark")
    print("=" * 70)

    for fwhm in [2.0, 3.0, 4.0]:
        print(f"\n── FWHM = {fwhm:.1f} px ({fwhm * 1.01:.2f} arcsec) ──")
        img, sources, _ = make_synthetic_image(fwhm)
        img_sub, bkg_rms = sky_subtract(img)
        snr = sources["snr_true"]
        print(f"  Sources: {len(sources)}, SNR range: {snr.min():.1f}–{snr.max():.1f}")

        approaches = {}
        approaches["Aperture"] = approach_aperture(img_sub, sources, fwhm, bkg_rms)
        approaches["Moffat"] = approach_moffat(img_sub, sources, fwhm, bkg_rms)
        approaches["PSFEx (perfect)"] = approach_psfex_stamp(img_sub, sources, fwhm, bkg_rms, 0.0)
        approaches["PSFEx (+0.3px)"] = approach_psfex_stamp(img_sub, sources, fwhm, bkg_rms, 0.3)

        target_src = sources[N_STARS_FOR_EPSF:]
        epsf_flux, epsf_t = approach_epsf(img_sub, sources, fwhm, bkg_rms)
        approaches["ePSF"] = (
            (epsf_flux - target_src["flux_true"]) / target_src["flux_true"],
            epsf_t,
            target_src["snr_true"],
        )

        print(f"\n  {'Method':<22} {'Med|bias|% SNR>10':>18} {'SNR>30':>8} {'ms':>8}")
        print(f"  {'-'*60}")

        for name in ["Aperture", "Moffat", "PSFEx (perfect)", "PSFEx (+0.3px)"]:
            flux_fit, t_ms = approaches[name]
            frac_err = (flux_fit - sources["flux_true"]) / sources["flux_true"]
            b10, _ = _stats(frac_err, snr, 10)
            b30, _ = _stats(frac_err, snr, 30)
            print(f"  {name:<22} {b10:>17.2f}% {b30:>7.2f}% {t_ms:>7.1f}")

        frac_epsf, t_epsf, snr_epsf = approaches["ePSF"]
        b10, _ = _stats(frac_epsf, snr_epsf, 10)
        b30, _ = _stats(frac_epsf, snr_epsf, 30)
        print(f"  {'ePSF (incl. build)':<22} {b10:>17.2f}% {b30:>7.2f}% {t_epsf:>7.1f}")

    print("\n" + "=" * 70)
    print(
        """
Conclusion:
  - PSFEx stamp with perfect PSF: ~0.5% bias (best accuracy, needs correct PSF)
  - Moffat analytic: ~1% bias (zero extra data, always available)
  - PSFEx +0.3px FWHM mismatch: ~7% bias (motivates spatially-varying PSF)
  - ePSF: inaccurate at low oversampling (EPSFBuilder normalization issue)

ztforce uses the sciimgdao.psf DAOPhot sidecar (spatially varying, 88KB,
available for 100% of ZTF science images) with a matched-filter estimator.
    """
    )


if __name__ == "__main__":
    main()
