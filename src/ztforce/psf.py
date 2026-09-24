"""DAOPhot PSF sidecar parsing and forced PSF photometry at a fixed position."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord

from .exceptions import PSFBuildError, WCSError
from .image import ZTFImage
from .utils import annular_background, flux_to_ab_mag, has_nan_nearby

# Annular sky background, in pixels beyond the PSF radius: just outside the region the
# PSF model covers, so the star's own wings are not taken as sky.  A closer annulus
# (2-4 FWHM) biased blank-sky fluxes positive by ~0.4 sigma.
_SKY_ANNULUS_GAP_PX = 1
_SKY_ANNULUS_WIDTH_PX = 8


def parse_daophot_psf(psf_fpath: str | Path) -> dict:
    """Parse a ZTF DAOPhot PSF sidecar file (sciimgdao.psf).

    The file format follows the DAOPHOT convention (Stetson 1987, PASP, 99, 191):
    a Gaussian analytic base plus spatially-varying lookup-table residuals.

    Returns a dict with keys ``psf_type``, ``psf_size``, ``n_tables``,
    ``norm_factor``, ``x_cen``, ``y_cen``, ``sigmas`` (the Gaussian's half-widths at
    half-maximum in x and y, despite the name), ``tables`` (sampled every half pixel).
    Pass the result to :func:`reconstruct_psf` to get a normalised PSF stamp.
    """
    with open(psf_fpath) as f:
        lines = f.readlines()

    hdr = lines[0].split()
    try:
        psf_type = hdr[0]
        psf_size = int(hdr[1])
        n_tables = int(hdr[3])
        # hdr[6] = normalization factor (peak amplitude of analytic Gaussian base)
        # hdr[7], hdr[8] = image center (x, y)
        norm_factor = float(hdr[6])
        x_cen = float(hdr[7])
        y_cen = float(hdr[8])
        sigmas = [float(v) for v in lines[1].split()]
    except (IndexError, ValueError) as exc:
        raise PSFBuildError(f"Malformed PSF header in {psf_fpath}: {exc}") from exc

    # Fixed-width scientific notation: adjacent negatives lack a space delimiter
    all_vals: list[float] = []
    for line in lines[2:]:
        tokens = re.findall(r"[+-]?\d+\.\d+E[+-]\d+", line)
        all_vals.extend(float(t) for t in tokens)

    expected = n_tables * psf_size * psf_size
    if len(all_vals) != expected:
        raise PSFBuildError(f"Expected {expected} PSF table values, got {len(all_vals)} in {psf_fpath}.")

    tables = np.array(all_vals).reshape(n_tables, psf_size, psf_size)
    return dict(
        psf_type=psf_type,
        psf_size=psf_size,
        n_tables=n_tables,
        norm_factor=norm_factor,
        x_cen=x_cen,
        y_cen=y_cen,
        sigmas=sigmas,
        tables=tables,
    )


def psf_radius(parsed: dict) -> int:
    """Radius in image pixels of the region the PSF lookup tables cover.

    DAOPHOT samples its tables at half-pixel spacing, ``NPSF = 2*(NINT(2*PSFRAD)+1)+1``
    (``psf.f``), so a 47x47 table spans 23x23 image pixels, radius 11.
    """
    return (parsed["psf_size"] - 1) // 4


_erf = np.vectorize(math.erf, otypes=[float])
_LN2 = math.log(2.0)


def _gauss_pixel_integral(d: np.ndarray, hwhm: float) -> np.ndarray:
    """Integral of exp(-ln2 (x/hwhm)^2) over the pixel [d - 0.5, d + 0.5].

    DAOPHOT's DAOERF: the Gaussian is parameterised by its half-width at half-maximum
    and integrated over each pixel rather than sampled at its centre.
    """
    sigma = hwhm / math.sqrt(2.0 * _LN2)
    k = math.sqrt(2.0) * sigma
    return sigma * math.sqrt(math.pi / 2.0) * (_erf((d + 0.5) / k) - _erf((d - 0.5) / k))


def _catmull_rom(f1, f2, f3, f4, t):
    """Cubic through f2..f3 at fraction t, as in DAOPHOT's BICUBC."""
    c1 = 0.5 * (f3 - f1)
    c2 = 3.0 * (f3 - f2 - c1) - 0.5 * (f4 - f2) + c1
    c3 = f3 - f2 - c1 - c2
    return ((c3 * t + c2) * t + c1) * t + f2


def _bicubic(table: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Interpolate ``table[row, col]`` at fractional (col=u, row=v), DAOPHOT BICUBC style.

    Points whose 4x4 neighbourhood leaves the table get 0.
    """
    n = table.shape[0]
    lx, ly = np.floor(u).astype(int), np.floor(v).astype(int)
    tx, ty = u - lx, v - ly

    def at(r, c):
        ok = (r >= 0) & (r < n) & (c >= 0) & (c < n)
        return np.where(ok, table[np.clip(r, 0, n - 1), np.clip(c, 0, n - 1)], 0.0)

    r0, r1, r2, r3 = (
        _catmull_rom(at(ly + j, lx - 1), at(ly + j, lx), at(ly + j, lx + 1), at(ly + j, lx + 2), tx)
        for j in (-1, 0, 1, 2)
    )
    return _catmull_rom(r0, r1, r2, r3, ty)


def reconstruct_psf(
    parsed: dict, x_target: float, y_target: float, frac_x: float = 0.0, frac_y: float = 0.0
) -> np.ndarray:
    """Reconstruct the normalized PSF stamp for a star at image position (x_target, y_target).

    The stamp is (2R+1, 2R+1), R = :func:`psf_radius`, centred on the pixel nearest the
    star; ``frac_x``/``frac_y`` (in [-0.5, 0.5]) are the star's offset from that pixel's
    centre, so the model is evaluated at each pixel's true distance from the star.
    Pixels beyond R are zero, as DAOPHOT only defines the PSF within PSFRAD.  The
    stamp is normalized to sum=1.
    """
    sigmas = parsed["sigmas"]
    tables = parsed["tables"]
    norm_factor = parsed["norm_factor"]
    x_cen = parsed["x_cen"]
    y_cen = parsed["y_cen"]

    r = psf_radius(parsed)
    mid = (parsed["psf_size"] - 1) // 2  # table centre, 0-based
    ky, kx = np.mgrid[-r : r + 1, -r : r + 1]
    ddx, ddy = kx - frac_x, ky - frac_y  # each pixel's offset from the star

    # Analytic Gaussian base (DAOPHOT PROFIL, type GAUSSIAN): ``sigmas`` holds the
    # half-widths at half-maximum in x and y, and the profile is pixel-integrated.
    hx, hy = sigmas[0], sigmas[1]
    gauss = norm_factor * _gauss_pixel_integral(ddx, hx) * _gauss_pixel_integral(ddy, hy) / (hx * hy)

    # Normalized position offsets in [-1, 1]
    dx = (x_target - x_cen) / x_cen
    dy = (y_target - y_cen) / y_cen

    # Lookup-table residuals, sampled every half pixel (DAOPHOT USEPSF: XX = 2*DX + MIDDLE)
    # and interpolated bicubically between entries.
    u, v = 2.0 * ddx + mid, 2.0 * ddy + mid
    weights = _poly_weights(dx, dy, parsed["n_tables"])
    residual = sum(w * _bicubic(t, u, v) for w, t in zip(weights, tables, strict=False))

    psf = gauss + residual
    psf = np.where(np.hypot(ddx, ddy) <= r, np.clip(psf, 0.0, None), 0.0)
    total = psf.sum()
    if total == 0:
        raise PSFBuildError("PSF reconstruction produced an all-zero stamp.")
    return psf / total


def _poly_weights(dx: float, dy: float, n: int) -> list[float]:
    """Return polynomial basis weights for n lookup tables.

    Follows DAOPHOT's USEPSF (Stetson 1987, PASP, 99, 191):
      n=1: [1]
      n=3: [1, dx, dy]
      n=6: [1, dx, dy, 1.5 dx^2 - 0.5, dx*dy, 1.5 dy^2 - 0.5]
    """
    basis = [1.0, dx, dy, 1.5 * dx * dx - 0.5, dx * dy, 1.5 * dy * dy - 0.5]
    return basis[:n]


def forced_phot_at_position(
    image: ZTFImage,
    parsed_psf: dict,
    target_coord: SkyCoord,
) -> dict:
    """Measure forced PSF photometry at a fixed sky position.

    Only the amplitude is free; position is locked.  Uses the optimal
    matched-filter estimator (Naylor 1998, MNRAS, 296, 339):
    ``flux = Σ(data·psf/σ²) / Σ(psf²/σ²)``.

    Returns a dict with keys ``flux``, ``flux_err``, ``mag``, ``mag_err``,
    ``chisq``, ``flags``, ``x_fit``, ``y_fit``.  ``chisq`` is the reduced chi-squared
    of the fit over the PSF footprint.  ``flags=1`` means the position was too
    close to the image edge or a NaN region.
    """
    nan_result = dict(
        flux=float("nan"),
        flux_err=float("nan"),
        mag=float("nan"),
        mag_err=float("nan"),
        chisq=float("nan"),
        flags=1,
        x_fit=float("nan"),
        y_fit=float("nan"),
    )

    try:
        x0, y0 = image.sky_to_pixel(target_coord)
        x0_full, y0_full = image.sky_to_full_quadrant_pixel(target_coord)
    except WCSError:
        return nan_result

    # Integer center pixel (cutout-local for array indexing)
    xi, yi = int(round(x0)), int(round(y0))
    half = psf_radius(parsed_psf)  # fit box: the region the PSF model covers
    sky_inner = half + _SKY_ANNULUS_GAP_PX
    sky_half = sky_inner + _SKY_ANNULUS_WIDTH_PX  # box holding the sky annulus
    ny, nx = image.data.shape

    # Reject if too close to edge
    if xi - sky_half < 0 or xi + sky_half + 1 > nx or yi - sky_half < 0 or yi + sky_half + 1 > ny:
        return nan_result

    # Reject if any NaN within PSF footprint
    if has_nan_nearby(yi, xi, half, image.nan_mask):
        return nan_result

    # Estimate the local sky from an annulus, then cut out the fit box
    sky_box = image.data[yi - sky_half : yi + sky_half + 1, xi - sky_half : xi + sky_half + 1]
    sky_level, sky_rms = annular_background(
        sky_box,
        float(sky_half),
        float(sky_half),
        float(sky_inner),
        float(sky_half),
    )
    raw_cutout = image.data[yi - half : yi + half + 1, xi - half : xi + half + 1]
    cutout = raw_cutout - sky_level

    # PSF model uses full-quadrant coordinates for the spatially-varying polynomial
    psf_stamp = reconstruct_psf(parsed_psf, x0_full, y0_full, x0 - xi, y0 - yi)

    # Noise model: Poisson + sky background variance
    fallback_var = max(sky_rms**2, 1.0)
    noise_var = sky_rms**2 + np.abs(cutout) / image.gain
    noise_var = np.where(noise_var > 0, noise_var, fallback_var)

    # Matched-filter flux estimator (optimal for Gaussian noise)
    w = psf_stamp / noise_var
    denom = (psf_stamp * w).sum()
    if denom <= 0:
        return nan_result

    flux = (cutout * w).sum() / denom
    flux_var = 1.0 / denom
    flux_err = float(np.sqrt(flux_var))
    flux = float(flux)
    # Reduced chi-squared of the amplitude-only fit (one free parameter).
    chisq = float((((cutout - flux * psf_stamp) ** 2) / noise_var).sum() / (cutout.size - 1))

    mag, mag_err = flux_to_ab_mag(flux, image.zero_point, flux_err)

    return dict(
        flux=flux,
        flux_err=flux_err,
        mag=float(mag) if mag is not None else float("nan"),
        mag_err=float(mag_err) if mag_err is not None else float("nan"),
        chisq=chisq,
        flags=0,
        x_fit=x0,
        y_fit=y0,
    )
