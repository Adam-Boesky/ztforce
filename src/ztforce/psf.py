"""DAOPhot PSF sidecar parsing and forced PSF photometry at a fixed position."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord

from .exceptions import PSFBuildError, WCSError
from .image import ZTFImage
from .utils import annular_background, flux_to_ab_mag, has_nan_nearby

# Annular sky background: inner/outer radius as multiples of the PSF FWHM
_SKY_ANNULUS_INNER_FWHM = 2.0
_SKY_ANNULUS_OUTER_FWHM = 4.0


def parse_daophot_psf(psf_fpath: str | Path) -> dict:
    """Parse a ZTF DAOPhot PSF sidecar file (sciimgdao.psf).

    The file format follows the DAOPHOT convention (Stetson 1987, PASP, 99, 191):
    a Gaussian analytic base plus spatially-varying lookup-table residuals.

    Returns a dict with keys ``psf_type``, ``psf_size``, ``n_tables``,
    ``norm_factor``, ``x_cen``, ``y_cen``, ``sigmas``, ``tables``.
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


def reconstruct_psf(parsed: dict, x_target: float, y_target: float) -> np.ndarray:
    """Reconstruct the normalized PSF stamp at image position (x_target, y_target).

    Returns a 2D array of shape (psf_size, psf_size) normalized to sum=1.
    """
    s = parsed["psf_size"]
    sigmas = parsed["sigmas"]
    tables = parsed["tables"]
    norm_factor = parsed["norm_factor"]
    x_cen = parsed["x_cen"]
    y_cen = parsed["y_cen"]

    c = s // 2
    row, col = np.mgrid[0:s, 0:s]

    # Analytic Gaussian base with peak = norm_factor
    gauss = norm_factor * np.exp(-0.5 * ((col - c) ** 2 / sigmas[0] ** 2 + (row - c) ** 2 / sigmas[1] ** 2))

    # Normalized position offsets in [-1, 1]
    dx = (x_target - x_cen) / x_cen
    dy = (y_target - y_cen) / y_cen

    # Polynomial basis for spatial variation: [1, dx, dy] (matches 3-table DAOPhot files)
    weights = _poly_weights(dx, dy, parsed["n_tables"])
    residual = sum(w * t for w, t in zip(weights, tables, strict=False))

    psf = gauss + residual
    psf = np.clip(psf, 0.0, None)
    total = psf.sum()
    if total == 0:
        raise PSFBuildError("PSF reconstruction produced an all-zero stamp.")
    return psf / total


def _poly_weights(dx: float, dy: float, n: int) -> list[float]:
    """Return polynomial basis weights for n lookup tables.

    Follows the DAOPHOT spatial-variation convention (Stetson 1987, PASP, 99, 191):
      n=1: [1]
      n=3: [1, dx, dy]
      n=6: [1, dx, dy, dx^2, dx*dy, dy^2]
    """
    if n == 1:
        return [1.0]
    if n == 3:
        return [1.0, dx, dy]
    if n == 6:
        return [1.0, dx, dy, dx * dx, dx * dy, dy * dy]
    # Generic: fill as many terms as available from the degree-2 expansion
    basis = [1.0, dx, dy, dx * dx, dx * dy, dy * dy]
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
    ``flags``, ``x_fit``, ``y_fit``.  ``flags=1`` means the position was too
    close to the image edge or a NaN region.
    """
    nan_result = dict(
        flux=float("nan"),
        flux_err=float("nan"),
        mag=float("nan"),
        mag_err=float("nan"),
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
    psf_size = parsed_psf["psf_size"]
    half = psf_size // 2
    ny, nx = image.data.shape

    # Reject if too close to edge
    if xi - half < 0 or xi + half + 1 > nx or yi - half < 0 or yi + half + 1 > ny:
        return nan_result

    # Reject if any NaN within PSF footprint
    if has_nan_nearby(yi, xi, half, image.nan_mask):
        return nan_result

    # Extract raw cutout; estimate and subtract local sky from an annulus
    raw_cutout = image.data[yi - half : yi + half + 1, xi - half : xi + half + 1].copy()
    sky_level, sky_rms = annular_background(
        raw_cutout,
        float(half),
        float(half),
        _SKY_ANNULUS_INNER_FWHM * image.fwhm,
        _SKY_ANNULUS_OUTER_FWHM * image.fwhm,
    )
    cutout = raw_cutout - sky_level

    # PSF model uses full-quadrant coordinates for the spatially-varying polynomial
    psf_stamp = reconstruct_psf(parsed_psf, x0_full, y0_full)

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

    mag, mag_err = flux_to_ab_mag(flux, image.zero_point, flux_err)

    return dict(
        flux=flux,
        flux_err=flux_err,
        mag=float(mag) if mag is not None else float("nan"),
        mag_err=float(mag_err) if mag_err is not None else float("nan"),
        flags=0,
        x_fit=x0,
        y_fit=y0,
    )
