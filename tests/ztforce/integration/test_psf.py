"""Tests for ztforce.psf (DAOPhot PSF parsing and forced photometry)."""

from __future__ import annotations

import numpy as np
import pytest

# ── parse_daophot_psf ─────────────────────────────────────────────────────────


def test_parse_returns_expected_keys(synthetic_psf_file):
    """parse_daophot_psf returns all required dict keys."""
    from ztforce.psf import parse_daophot_psf

    parsed = parse_daophot_psf(synthetic_psf_file)
    for key in ("psf_type", "psf_size", "n_tables", "norm_factor", "x_cen", "y_cen", "sigmas", "tables"):
        assert key in parsed


def test_parse_table_shape(synthetic_psf_file):
    """tables array has shape (n_tables, psf_size, psf_size)."""
    from ztforce.psf import parse_daophot_psf

    parsed = parse_daophot_psf(synthetic_psf_file)
    s = parsed["psf_size"]
    n = parsed["n_tables"]
    assert parsed["tables"].shape == (n, s, s)


def test_parse_header_values(synthetic_psf_file):
    """Header metadata is parsed to the correct numeric values."""
    from ztforce.psf import parse_daophot_psf

    parsed = parse_daophot_psf(synthetic_psf_file)
    assert parsed["psf_type"] == "GAUSSIAN"
    assert parsed["psf_size"] == 11
    assert parsed["n_tables"] == 3
    assert parsed["norm_factor"] == pytest.approx(1000.0)
    assert parsed["x_cen"] == pytest.approx(1535.5)
    assert parsed["y_cen"] == pytest.approx(1539.5)


def test_parse_handles_adjacent_negative_numbers(tmp_path):
    """Regex tokenizer handles DAOPhot's fixed-width format with adjacent negatives."""
    from ztforce.psf import parse_daophot_psf

    # Hand-craft a line with two adjacent negatives (no space between)
    header = " GAUSSIAN    3    2    1    0   14.000    1000.000  1535.5  1539.5\n"
    sigmas = "  1.500000E+00 1.500000E+00\n"
    # 9 values: 3 per row, 3 rows — include an adjacent-negative pair
    data_line = "  1.000000E+00-2.000000E+00  3.000000E+00\n" * 3

    path = tmp_path / "adj.psf"
    path.write_text(header + sigmas + data_line)

    parsed = parse_daophot_psf(path)
    flat = parsed["tables"][0].flatten()
    assert flat[1] == pytest.approx(-2.0)


def test_parse_bad_file_raises_psf_build_error(tmp_path):
    """PSFBuildError is raised for a malformed sidecar file."""
    from ztforce.exceptions import PSFBuildError
    from ztforce.psf import parse_daophot_psf

    (tmp_path / "bad.psf").write_text("GARBAGE\n1.0 2.0\n")
    with pytest.raises(PSFBuildError):
        parse_daophot_psf(tmp_path / "bad.psf")


def test_parse_wrong_value_count_raises(tmp_path):
    """PSFBuildError is raised when the table has fewer values than expected."""
    from ztforce.exceptions import PSFBuildError
    from ztforce.psf import parse_daophot_psf

    # Valid header but only 1 data value instead of n_tables * psf_size^2
    path = tmp_path / "short.psf"
    path.write_text(
        " GAUSSIAN    5    2    1    0   14.000    1000.000  1535.5  1539.5\n"
        "  1.500000E+00 1.500000E+00\n"
        "  1.000000E+00\n"  # only 1 value, need 25
    )
    with pytest.raises(PSFBuildError):
        parse_daophot_psf(path)


# ── reconstruct_psf ───────────────────────────────────────────────────────────


def test_reconstruct_sums_to_one(synthetic_psf_file):
    """Reconstructed PSF normalizes to sum=1."""
    from ztforce.psf import parse_daophot_psf, reconstruct_psf

    parsed = parse_daophot_psf(synthetic_psf_file)
    psf = reconstruct_psf(parsed, parsed["x_cen"], parsed["y_cen"])
    assert abs(psf.sum() - 1.0) < 1e-6


def test_reconstruct_non_negative(synthetic_psf_file):
    """Reconstructed PSF has no negative values (clipped)."""
    from ztforce.psf import parse_daophot_psf, reconstruct_psf

    parsed = parse_daophot_psf(synthetic_psf_file)
    psf = reconstruct_psf(parsed, parsed["x_cen"], parsed["y_cen"])
    assert psf.min() >= 0.0


def test_reconstruct_peak_near_center(synthetic_psf_file):
    """For a symmetric PSF, the peak falls in the central pixel."""
    from ztforce.psf import parse_daophot_psf, reconstruct_psf

    parsed = parse_daophot_psf(synthetic_psf_file)
    psf = reconstruct_psf(parsed, parsed["x_cen"], parsed["y_cen"])
    c = parsed["psf_size"] // 2
    peak_r, peak_c = np.unravel_index(psf.argmax(), psf.shape)
    assert abs(peak_r - c) <= 1
    assert abs(peak_c - c) <= 1


def test_reconstruct_varies_with_position(synthetic_psf_file):
    """PSF stamp changes when reconstructed at different image positions."""
    from ztforce.psf import parse_daophot_psf, reconstruct_psf

    parsed = parse_daophot_psf(synthetic_psf_file)
    psf_center = reconstruct_psf(parsed, parsed["x_cen"], parsed["y_cen"])
    psf_corner = reconstruct_psf(parsed, 200.0, 200.0)
    assert not np.allclose(psf_center, psf_corner)


def test_poly_weights_n1():
    """_poly_weights for n=1 returns [1.0]."""
    from ztforce.psf import _poly_weights

    assert _poly_weights(0.5, -0.3, 1) == [1.0]


def test_poly_weights_n3():
    """_poly_weights for n=3 returns [1, dx, dy]."""
    from ztforce.psf import _poly_weights

    dx, dy = 0.5, -0.3
    assert _poly_weights(dx, dy, 3) == [1.0, dx, dy]


def test_poly_weights_n6():
    """_poly_weights for n=6 returns full degree-2 basis."""
    from ztforce.psf import _poly_weights

    dx, dy = 0.5, -0.3
    result = _poly_weights(dx, dy, 6)
    assert result == [1.0, dx, dy, dx * dx, dx * dy, dy * dy]


def test_reconstruct_all_zero_raises(tmp_path):
    """PSFBuildError is raised when reconstruction produces an all-zero stamp."""
    from ztforce.exceptions import PSFBuildError
    from ztforce.psf import parse_daophot_psf, reconstruct_psf

    # Build a PSF where gauss + residuals will be entirely negative → clipped to zero
    psf_size = 3
    norm_factor = -1000.0  # negative norm → Gaussian is negative everywhere
    header = f" GAUSSIAN  {psf_size:3d}    2    1    0   14.000  {norm_factor:12.3f}  1535.5  1539.5\n"
    sigmas_line = "  1.500000E+00 1.500000E+00\n"
    zeros = "  0.000000E+00 " * (psf_size * psf_size)

    path = tmp_path / "allzero.psf"
    path.write_text(header + sigmas_line + zeros + "\n")

    parsed = parse_daophot_psf(path)
    with pytest.raises(PSFBuildError, match="all-zero"):
        reconstruct_psf(parsed, 1535.5, 1539.5)


def test_poly_weights_generic_n4():
    """_poly_weights for n=4 falls back to the first 4 degree-2 basis terms."""
    from ztforce.psf import _poly_weights

    dx, dy = 0.5, -0.3
    result = _poly_weights(dx, dy, 4)
    assert result == [1.0, dx, dy, dx * dx]


def test_poly_weights_generic_n5():
    """_poly_weights for n=5 falls back to the first 5 degree-2 basis terms."""
    from ztforce.psf import _poly_weights

    dx, dy = 0.5, -0.3
    result = _poly_weights(dx, dy, 5)
    assert result == [1.0, dx, dy, dx * dx, dx * dy]


# ── forced_phot_at_position ───────────────────────────────────────────────────


def test_forced_phot_detects_injected_source(synthetic_fits_file, synthetic_psf_file, mock_config):
    """Matched-filter flux recovers total integrated flux of an injected Gaussian."""
    from ztforce.image import ZTFImage
    from ztforce.psf import forced_phot_at_position, parse_daophot_psf

    path, cx, cy = synthetic_fits_file
    img = ZTFImage(str(path), "g", mock_config)
    parsed_psf = parse_daophot_psf(synthetic_psf_file)
    # Set sigmas to match the injected source (FWHM=3px → sigma=3/2.355)
    parsed_psf["sigmas"] = [3.0 / 2.355, 3.0 / 2.355]

    coord = img.pixel_to_sky(float(cx), float(cy))
    result = forced_phot_at_position(img, parsed_psf, coord)

    assert result["flags"] == 0
    assert result["flux"] > 0
    # Matched filter → total integrated flux ≈ peak * 2π * σ²
    sigma = 3.0 / 2.355
    expected_flux = 5000.0 * 2 * np.pi * sigma**2
    frac_err = abs(result["flux"] - expected_flux) / expected_flux
    assert frac_err < 0.30, f"Flux {result['flux']:.0f} vs expected {expected_flux:.0f}"


def test_forced_phot_returns_finite_mag(synthetic_fits_file, synthetic_psf_file, mock_config):
    """A detected source has a finite magnitude."""
    from ztforce.image import ZTFImage
    from ztforce.psf import forced_phot_at_position, parse_daophot_psf

    path, cx, cy = synthetic_fits_file
    img = ZTFImage(str(path), "g", mock_config)
    parsed_psf = parse_daophot_psf(synthetic_psf_file)
    parsed_psf["sigmas"] = [3.0 / 2.355, 3.0 / 2.355]

    coord = img.pixel_to_sky(float(cx), float(cy))
    result = forced_phot_at_position(img, parsed_psf, coord)

    assert np.isfinite(result["mag"])
    assert np.isfinite(result["mag_err"])


def test_forced_phot_edge_returns_flags1(synthetic_fits_file, synthetic_psf_file, mock_config):
    """Position within PSF half-width of image edge returns flags=1 and NaN flux."""
    from ztforce.image import ZTFImage
    from ztforce.psf import forced_phot_at_position, parse_daophot_psf

    path, _, _ = synthetic_fits_file
    img = ZTFImage(str(path), "g", mock_config)
    parsed_psf = parse_daophot_psf(synthetic_psf_file)

    edge_coord = img.pixel_to_sky(1.0, 1.0)
    result = forced_phot_at_position(img, parsed_psf, edge_coord)

    assert result["flags"] == 1
    assert np.isnan(result["flux"])
    assert np.isnan(result["mag"])


def test_forced_phot_nan_region_returns_flags1(tmp_path, synthetic_psf_file, mock_config):
    """Position overlapping a NaN region returns flags=1."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from ztforce.image import ZTFImage
    from ztforce.psf import forced_phot_at_position, parse_daophot_psf

    size = 64
    cx, cy = 32, 32
    data = np.ones((size, size), dtype=np.float32) * 100.0
    # Put NaN right at the target position
    data[cy, cx] = float("nan")

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [cx + 1, cy + 1]
    wcs.wcs.cdelt = [-0.000281, 0.000281]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdr = wcs.to_header()
    hdr.update(MAGZP=26.3, OBSJD=2459000.0, GAIN=6.2, MEDFWHM=3.0, RADESYS="ICRS")

    path = tmp_path / "nan.fits"
    fits.writeto(str(path), data, hdr)

    img = ZTFImage(str(path), "g", mock_config)
    parsed_psf = parse_daophot_psf(synthetic_psf_file)

    coord = img.pixel_to_sky(float(cx), float(cy))
    result = forced_phot_at_position(img, parsed_psf, coord)
    assert result["flags"] == 1
