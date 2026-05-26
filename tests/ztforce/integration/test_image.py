"""Tests for ztforce.image (ZTFImage)."""

import numpy as np
import pytest


def test_header_loaded(synthetic_fits_file, mock_config):
    """ZTFImage.header returns the FITS header."""
    from ztforce.image import ZTFImage

    path, _, _ = synthetic_fits_file
    img = ZTFImage(str(path), "g", mock_config)
    assert "MAGZP" in img.header
    assert "OBSJD" in img.header


def test_data_shape(synthetic_fits_file, mock_config):
    """ZTFImage.data returns a 2D float64 array."""
    from ztforce.image import ZTFImage

    path, _, _ = synthetic_fits_file
    img = ZTFImage(str(path), "g", mock_config)
    assert img.data.ndim == 2
    assert img.data.dtype == np.float64


def test_data_is_native_endian(synthetic_fits_file, mock_config):
    """data array has native byte order (required by SEP)."""
    from ztforce.image import ZTFImage

    path, _, _ = synthetic_fits_file
    img = ZTFImage(str(path), "g", mock_config)
    assert img.data.flags["C_CONTIGUOUS"]
    # Native endian: byte order must not be big- or little-endian (> or <)
    assert img.data.dtype.byteorder not in (">", "<")


def test_scalar_properties(synthetic_fits_file, mock_config):
    """gain, fwhm, zero_point, obs_jd, mag_limit are read from header."""
    from ztforce.image import ZTFImage

    path, _, _ = synthetic_fits_file
    img = ZTFImage(str(path), "g", mock_config)
    assert img.gain == pytest.approx(6.2)
    assert img.fwhm == pytest.approx(3.0)
    assert img.zero_point == pytest.approx(26.3)
    assert img.obs_jd == pytest.approx(2459000.0)
    assert img.mag_limit == pytest.approx(21.0)


def test_gain_fallback_nframes(tmp_path, mock_config):
    """gain falls back to 5.8 * NFRAMES when GAIN header is absent."""
    from astropy.io import fits
    from ztforce.image import ZTFImage

    path = tmp_path / "nframes.fits"
    hdr = fits.Header()
    hdr["NFRAMES"] = 3
    hdr["MAGZP"] = 26.3
    hdr["OBSJD"] = 2459000.0
    hdr["MEDFWHM"] = 3.0
    hdr["RADESYS"] = "ICRS"
    fits.writeto(str(path), np.zeros((64, 64), dtype=np.float32), hdr)

    img = ZTFImage(str(path), "g", mock_config)
    assert img.gain == pytest.approx(5.8 * 3)


def test_gain_fallback_default(tmp_path, mock_config):
    """gain falls back to config.default_gain when no GAIN or NFRAMES header."""
    from astropy.io import fits
    from ztforce.image import ZTFImage

    path = tmp_path / "nogain.fits"
    hdr = fits.Header()
    hdr["MAGZP"] = 26.3
    hdr["OBSJD"] = 2459000.0
    hdr["MEDFWHM"] = 3.0
    hdr["RADESYS"] = "ICRS"
    fits.writeto(str(path), np.zeros((64, 64), dtype=np.float32), hdr)

    img = ZTFImage(str(path), "g", mock_config)
    assert img.gain == pytest.approx(mock_config.default_gain)


def test_fwhm_falls_back_to_seeing(tmp_path, mock_config):
    """fwhm uses SEEING when MEDFWHM is absent."""
    from astropy.io import fits
    from ztforce.image import ZTFImage

    path = tmp_path / "seeing.fits"
    hdr = fits.Header()
    hdr["GAIN"] = 6.2
    hdr["MAGZP"] = 26.3
    hdr["OBSJD"] = 2459000.0
    hdr["SEEING"] = 2.5
    hdr["RADESYS"] = "ICRS"
    fits.writeto(str(path), np.zeros((64, 64), dtype=np.float32), hdr)

    img = ZTFImage(str(path), "g", mock_config)
    assert img.fwhm == pytest.approx(2.5)


def test_radecsys_rename(tmp_path, mock_config):
    """Old RADECSYS header keyword is renamed to RADESYS so WCS builds cleanly."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from ztforce.image import ZTFImage

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [33, 33]
    wcs.wcs.cdelt = [-0.000281, 0.000281]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdr = wcs.to_header()
    hdr["MAGZP"] = 26.3
    hdr["OBSJD"] = 2459000.0
    hdr["GAIN"] = 6.2
    hdr["MEDFWHM"] = 3.0
    # Add old keyword to trigger rename path
    hdr["RADECSYS"] = "ICRS"

    path = tmp_path / "radecsys.fits"
    fits.writeto(str(path), np.zeros((64, 64), dtype=np.float32), hdr)

    img = ZTFImage(str(path), "g", mock_config)
    # WCS should build without raising WCSError
    wcs_built = img.wcs
    assert wcs_built is not None


def test_wcs_sky_to_pixel_round_trip(synthetic_fits_file, mock_config):
    """sky_to_pixel and pixel_to_sky are inverses to within 0.01 px."""
    from ztforce.image import ZTFImage

    path, cx, cy = synthetic_fits_file
    img = ZTFImage(str(path), "g", mock_config)

    coord = img.pixel_to_sky(float(cx), float(cy))
    x2, y2 = img.sky_to_pixel(coord)
    assert abs(x2 - cx) < 0.01
    assert abs(y2 - cy) < 0.01


def test_nan_mask(synthetic_fits_file, mock_config):
    """nan_mask is True only where data is NaN."""
    from ztforce.image import ZTFImage

    path, cx, cy = synthetic_fits_file
    img = ZTFImage(str(path), "g", mock_config)
    # Synthetic image has no NaNs
    assert not img.nan_mask.any()


def test_footprint_returns_ra_dec_bounds(synthetic_fits_file, mock_config):
    """footprint() returns ((ra_min, ra_max), (dec_min, dec_max)) covering the image."""
    from ztforce.image import ZTFImage

    path, _, _ = synthetic_fits_file
    img = ZTFImage(str(path), "g", mock_config)
    (ra_min, ra_max), (dec_min, dec_max) = img.footprint()
    assert ra_min < ra_max
    assert dec_min < dec_max
    # Synthetic image is centered near (150, 2), so bounds should straddle those coords
    assert ra_min < 150.0 < ra_max
    assert dec_min < 2.0 < dec_max


def test_cutout_origin_defaults_to_zero(synthetic_fits_file, mock_config):
    """cutout_origin is (0, 0) when LTV keywords are absent (full-image download)."""
    from ztforce.image import ZTFImage

    path, _, _ = synthetic_fits_file
    img = ZTFImage(str(path), "g", mock_config)
    assert img.cutout_origin == (0.0, 0.0)


def test_cutout_origin_from_ltv_keywords(tmp_path, mock_config):
    """cutout_origin reflects LTV1/LTV2 using the IRAF negative-offset convention."""
    from astropy.io import fits
    from ztforce.image import ZTFImage

    hdr = fits.Header()
    hdr["MAGZP"] = 26.3
    hdr["OBSJD"] = 2459000.0
    hdr["GAIN"] = 6.2
    hdr["MEDFWHM"] = 3.0
    hdr["RADESYS"] = "ICRS"
    # IRSA convention: LTV = -(0-indexed start), so origin = -LTV
    hdr["LTV1"] = -512.0
    hdr["LTV2"] = -256.0

    path = tmp_path / "cutout.fits"
    fits.writeto(str(path), np.zeros((64, 64), dtype=np.float32), hdr)

    img = ZTFImage(str(path), "g", mock_config)
    assert img.cutout_origin == (512.0, 256.0)


def test_sky_to_full_quadrant_pixel_no_offset(synthetic_fits_file, mock_config):
    """sky_to_full_quadrant_pixel equals sky_to_pixel when there is no cutout offset."""
    from ztforce.image import ZTFImage

    path, cx, cy = synthetic_fits_file
    img = ZTFImage(str(path), "g", mock_config)
    coord = img.pixel_to_sky(float(cx), float(cy))
    x_cut, y_cut = img.sky_to_pixel(coord)
    x_full, y_full = img.sky_to_full_quadrant_pixel(coord)
    assert x_full == x_cut
    assert y_full == y_cut


def test_sky_to_full_quadrant_pixel_adds_offset(tmp_path, mock_config):
    """sky_to_full_quadrant_pixel adds cutout_origin to the cutout-local coordinates."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from ztforce.image import ZTFImage

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [33, 33]
    wcs.wcs.cdelt = [-0.000281, 0.000281]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdr = wcs.to_header()
    hdr.update(MAGZP=26.3, OBSJD=2459000.0, GAIN=6.2, MEDFWHM=3.0, RADESYS="ICRS")
    hdr["LTV1"] = -100.0
    hdr["LTV2"] = -200.0

    path = tmp_path / "offset.fits"
    fits.writeto(str(path), np.zeros((64, 64), dtype=np.float32), hdr)

    img = ZTFImage(str(path), "g", mock_config)
    coord = img.pixel_to_sky(10.0, 10.0)
    x_cut, y_cut = img.sky_to_pixel(coord)
    x_full, y_full = img.sky_to_full_quadrant_pixel(coord)
    assert abs(x_full - (x_cut + 100.0)) < 1e-6
    assert abs(y_full - (y_cut + 200.0)) < 1e-6


def test_radecsys_both_keywords_deletes_duplicate(tmp_path, mock_config):
    """When both RADECSYS and RADESYS are present, the duplicate RADECSYS is removed."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from ztforce.image import ZTFImage

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [33, 33]
    wcs.wcs.cdelt = [-0.000281, 0.000281]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdr = wcs.to_header()
    hdr["MAGZP"] = 26.3
    hdr["OBSJD"] = 2459000.0
    hdr["GAIN"] = 6.2
    hdr["MEDFWHM"] = 3.0
    # Both keywords present simultaneously — triggers the elif/del branch
    hdr["RADESYS"] = "ICRS"
    hdr["RADECSYS"] = "ICRS"

    path = tmp_path / "both.fits"
    fits.writeto(str(path), np.zeros((64, 64), dtype=np.float32), hdr)

    img = ZTFImage(str(path), "g", mock_config)
    assert "RADECSYS" not in img.header
    assert img.wcs is not None
