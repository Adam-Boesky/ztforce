"""Pure math helpers — no ztforce imports."""

from __future__ import annotations

import numpy as np


def flux_to_ab_mag(
    flux: float,
    zero_point: float,
    flux_err: float | None = None,
) -> tuple[float, float | None]:
    """Convert instrumental flux to AB magnitude.

    Uses the AB system definition from Oke & Gunn (1983, ApJ, 266, 713):
    ``mag = zero_point - 2.5 * log10(flux)``.

    Returns (mag, mag_err). mag_err is None when flux_err is not given.
    Returns (nan, nan) when flux <= 0.
    """
    if flux <= 0:
        nan = float("nan")
        return nan, (nan if flux_err is not None else None)
    mag = zero_point - 2.5 * np.log10(flux)
    if flux_err is None:
        return mag, None
    mag_err = abs(2.5 / np.log(10) * flux_err / flux)
    return mag, mag_err


def ab_mag_to_flux(
    mag: float,
    zero_point: float,
    mag_err: float | None = None,
) -> tuple[float, float | None]:
    """Convert AB magnitude to instrumental flux.

    Returns (flux, flux_err). flux_err is None when mag_err is not given.
    """
    flux = 10.0 ** ((zero_point - mag) / 2.5)
    if mag_err is None:
        return flux, None
    flux_err = abs(flux * np.log(10) / 2.5 * mag_err)
    return flux, flux_err


def snr_from_flux(flux: float, flux_err: float) -> float:
    """Signal-to-noise ratio from flux and its uncertainty."""
    if flux_err == 0:
        return float("inf")
    return flux / flux_err


def has_nan_nearby(
    row: int,
    col: int,
    radius: float,
    mask: np.ndarray,
) -> bool:
    """Return True if any pixel within *radius* of (row, col) is masked."""
    r0 = max(0, int(row - radius))
    r1 = min(mask.shape[0], int(row + radius) + 1)
    c0 = max(0, int(col - radius))
    c1 = min(mask.shape[1], int(col + radius) + 1)
    patch = mask[r0:r1, c0:c1]
    return bool(patch.any())


def nearest_odd_int(x: float) -> int:
    """Round *x* up to the nearest odd integer."""
    n = int(np.ceil(x))
    return n if n % 2 == 1 else n + 1


def annular_background(
    data: np.ndarray,
    cx: float,
    cy: float,
    r_inner: float,
    r_outer: float,
    sigma: float = 3.0,
) -> tuple[float, float]:
    """Sigma-clipped sky level and RMS in an annulus around (cx, cy).

    cx/cy follow the FITS/numpy column/row convention (cx = column index).
    Returns (sky_level, sky_rms).  Falls back to the global finite-pixel
    statistics when fewer than 5 annulus pixels are available.
    """
    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    annulus = (r2 >= r_inner**2) & (r2 <= r_outer**2) & np.isfinite(data)
    pixels = data[annulus]

    if len(pixels) < 5:
        finite = data[np.isfinite(data)]
        if len(finite) == 0:
            return 0.0, 0.0
        return float(np.median(finite)), float(np.std(finite))

    med = float(np.median(pixels))
    std = float(np.std(pixels))
    for _ in range(3):
        if std == 0:
            break
        keep = np.abs(pixels - med) < sigma * std
        if not keep.any():
            break
        pixels = pixels[keep]
        med = float(np.median(pixels))
        std = float(np.std(pixels))

    return med, std
