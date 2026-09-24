"""Tests for ztforce.utils."""

import numpy as np
import pytest

# ── flux_to_ab_mag ────────────────────────────────────────────────────────────


def test_flux_to_mag_known_value():
    """At zero_point=26.3, flux=1 gives mag=26.3."""
    from ztforce.utils import flux_to_ab_mag

    mag, _ = flux_to_ab_mag(1.0, 26.3)
    assert abs(mag - 26.3) < 1e-10


def test_flux_to_mag_matches_formula():
    """flux_to_ab_mag gives zp - 2.5 log10(flux) and 1.0857 * err / flux."""
    from ztforce.utils import flux_to_ab_mag

    for flux in [10.0, 1000.0, 1e5]:
        ferr = flux * 0.01
        mag, merr = flux_to_ab_mag(flux, 26.3, ferr)
        assert mag == pytest.approx(26.3 - 2.5 * np.log10(flux), rel=1e-12)
        assert merr == pytest.approx(2.5 / np.log(10) * 0.01, rel=1e-12)


def test_flux_to_mag_negative_flux_returns_nan():
    """Non-positive flux returns NaN magnitude."""
    from ztforce.utils import flux_to_ab_mag

    for bad_flux in [0.0, -1.0, -1e6]:
        mag, merr = flux_to_ab_mag(bad_flux, 26.3, 5.0)
        assert np.isnan(mag), f"Expected NaN for flux={bad_flux}"
        assert np.isnan(merr)


def test_flux_to_mag_no_error():
    """flux_to_ab_mag without flux_err returns None for mag_err."""
    from ztforce.utils import flux_to_ab_mag

    mag, merr = flux_to_ab_mag(1000.0, 26.3)
    assert np.isfinite(mag)
    assert merr is None


def test_mag_error_propagation():
    """Magnitude error follows sigma_m = 2.5/ln(10) * sigma_f/f."""
    from ztforce.utils import flux_to_ab_mag

    flux, ferr = 500.0, 50.0
    _, merr = flux_to_ab_mag(flux, 26.3, ferr)
    expected = 2.5 / np.log(10) * ferr / flux
    assert abs(merr - expected) < 1e-12


# ── has_nan_nearby ────────────────────────────────────────────────────────────


def test_has_nan_nearby_detects_masked_pixel():
    """Returns True when a masked pixel is within radius."""
    from ztforce.utils import has_nan_nearby

    mask = np.zeros((30, 30), dtype=bool)
    mask[15, 15] = True
    assert has_nan_nearby(15, 15, 1, mask)
    assert has_nan_nearby(14, 15, 2, mask)


def test_has_nan_nearby_clear_region():
    """Returns False when no masked pixel is within radius."""
    from ztforce.utils import has_nan_nearby

    mask = np.zeros((30, 30), dtype=bool)
    mask[15, 15] = True
    assert not has_nan_nearby(3, 3, 3, mask)


def test_has_nan_nearby_boundary():
    """Handles positions near the image edge without IndexError."""
    from ztforce.utils import has_nan_nearby

    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True
    assert has_nan_nearby(0, 0, 1, mask)
    assert not has_nan_nearby(9, 9, 1, mask)


# ── annular_background ────────────────────────────────────────────────────────


def test_annular_background_flat_sky():
    """Returns the correct sky level on a perfectly flat background."""
    from ztforce.utils import annular_background

    data = np.full((50, 50), 100.0)
    level, rms = annular_background(data, 25.0, 25.0, r_inner=5.0, r_outer=15.0)
    assert abs(level - 100.0) < 1e-6
    assert rms < 1e-6


def test_annular_background_clips_outliers():
    """Sigma clipping removes bright outliers from the sky estimate."""
    from ztforce.utils import annular_background

    rng = np.random.default_rng(42)
    data = rng.normal(50.0, 2.0, (60, 60))
    # Plant a bright spike at a known annulus position
    data[30, 40] = 1000.0
    level, _ = annular_background(data, 30.0, 30.0, r_inner=5.0, r_outer=15.0)
    assert abs(level - 50.0) < 1.0


def test_annular_background_excludes_source_core():
    """Sky level is unbiased even with a bright source inside r_inner."""
    from ztforce.utils import annular_background

    data = np.full((60, 60), 30.0)
    # Bright point source at center
    data[30, 30] = 50000.0
    level, _ = annular_background(data, 30.0, 30.0, r_inner=5.0, r_outer=15.0)
    assert abs(level - 30.0) < 1.0


def test_annular_background_fallback_on_sparse_annulus():
    """Falls back gracefully when fewer than 5 pixels are in the annulus."""
    from ztforce.utils import annular_background

    data = np.full((10, 10), 20.0)
    # Tiny image so inner/outer radii leave very few pixels
    level, _ = annular_background(data, 5.0, 5.0, r_inner=1.0, r_outer=1.5)
    assert np.isfinite(level)
