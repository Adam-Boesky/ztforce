"""Pure math helpers — no ztforce imports."""

from __future__ import annotations

import numpy as np


def flux_to_ab_mag(
    flux: float,
    zero_point: float,
    flux_err: float | None = None,
) -> tuple[float, float | None]:
    """Convert instrumental flux to AB magnitude.

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
