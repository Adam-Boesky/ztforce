"""Unit tests for ztforce modules. All tests are offline (no network calls)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ── exceptions ───────────────────────────────────────────────────────────────


def test_exception_hierarchy():
    """Every specific exception is a subclass of ZTForceError."""
    from ztforce.exceptions import (
        ConfigError,
        FITSDownloadError,
        NoImagesFoundError,
        PSFBuildError,
        WCSError,
        ZTForceError,
    )

    for cls in [ConfigError, FITSDownloadError, NoImagesFoundError, PSFBuildError, WCSError]:
        assert issubclass(cls, ZTForceError)
        with pytest.raises(ZTForceError):
            raise cls("test")


# ── utils ────────────────────────────────────────────────────────────────────


def test_flux_to_mag_round_trip():
    """flux_to_ab_mag and ab_mag_to_flux are exact inverses of each other."""
    from ztforce.utils import ab_mag_to_flux, flux_to_ab_mag

    flux, ferr = 5000.0, 50.0
    mag, merr = flux_to_ab_mag(flux, 26.3, ferr)
    flux2, ferr2 = ab_mag_to_flux(mag, 26.3, merr)
    assert abs(flux2 - flux) < 0.001
    assert abs(ferr2 - ferr) < 0.001


def test_flux_to_mag_negative_flux():
    """Negative flux returns NaN magnitude."""
    from ztforce.utils import flux_to_ab_mag

    mag, merr = flux_to_ab_mag(-10.0, 26.3, 5.0)
    assert np.isnan(mag)
    assert np.isnan(merr)


def test_has_nan_nearby():
    """has_nan_nearby detects masked pixels within the specified radius."""
    from ztforce.utils import has_nan_nearby

    mask = np.zeros((20, 20), dtype=bool)
    mask[10, 10] = True
    assert has_nan_nearby(10, 10, 1, mask)
    assert has_nan_nearby(9, 10, 2, mask)
    assert not has_nan_nearby(3, 3, 2, mask)


def test_nearest_odd_int():
    """nearest_odd_int always returns an odd integer >= x."""
    from ztforce.utils import nearest_odd_int

    assert nearest_odd_int(14.0) == 15
    assert nearest_odd_int(15.0) == 15
    assert nearest_odd_int(13.1) == 15
    assert nearest_odd_int(1.0) == 1


# ── config ────────────────────────────────────────────────────────────────────


def test_config_from_env(monkeypatch, tmp_path):
    """Environment variables are the second-priority credential source."""
    monkeypatch.setenv("ZTFORCE_IRSA_USER", "testuser")
    monkeypatch.setenv("ZTFORCE_IRSA_PASS", "testpass")
    from ztforce.config import build_config

    cfg = build_config(config_path=tmp_path / "nonexistent.toml")
    assert cfg.irsa_user == "testuser"
    assert cfg.irsa_pass == "testpass"


def test_config_direct_params(tmp_path):
    """Direct parameters override every other credential source."""
    from ztforce.config import build_config

    cfg = build_config(
        irsa_user="u",
        irsa_pass="p",
        config_path=tmp_path / "nonexistent.toml",
    )
    assert cfg.irsa_user == "u"
    assert cfg.irsa_pass == "p"


def test_config_error_when_no_credentials(monkeypatch, tmp_path):
    """ConfigError is raised when no credential source provides credentials."""
    monkeypatch.delenv("ZTFORCE_IRSA_USER", raising=False)
    monkeypatch.delenv("ZTFORCE_IRSA_PASS", raising=False)
    from ztforce.config import build_config
    from ztforce.exceptions import ConfigError

    with pytest.raises(ConfigError):
        build_config(config_path=tmp_path / "nonexistent.toml")


def test_config_from_toml(tmp_path):
    """Credentials in a TOML config file are read correctly."""
    toml_text = "[credentials]\nirsa_user = 'tomluser'\nirsa_pass = 'tomlpass'\n"
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(toml_text)
    from ztforce.config import build_config

    cfg = build_config(config_path=cfg_path)
    assert cfg.irsa_user == "tomluser"


def test_config_direct_overrides_env(monkeypatch, tmp_path):
    """Direct irsa_user/pass argument takes priority over environment variables."""
    monkeypatch.setenv("ZTFORCE_IRSA_USER", "envuser")
    monkeypatch.setenv("ZTFORCE_IRSA_PASS", "envpass")
    from ztforce.config import build_config

    cfg = build_config(irsa_user="direct", irsa_pass="direct2", config_path=tmp_path / "nonexistent.toml")
    assert cfg.irsa_user == "direct"


# ── cache ────────────────────────────────────────────────────────────────────


def test_cache_path_construction(tmp_path):
    """Cache paths encode field/ccdid/qid/band/obsjd deterministically."""
    from ztforce.cache import fits_path, lightcurve_path, make_cache, psf_path

    cache = make_cache(tmp_path)
    p = fits_path(cache, 468, 3, 2, "g", 2459271.12345)
    assert p.suffix == ".fits"
    assert "000468" in str(p)

    p2 = psf_path(cache, 468, 3, 2, "g", 2459271.12345)
    assert p2.suffix == ".psf"

    p3 = lightcurve_path(cache, 150.0, 2.0, "g")
    assert p3.suffix == ".ecsv"
    assert "150.00000" in str(p3)


def test_cache_directories_created(tmp_path):
    """Cache subdirectories are created lazily on first path request."""
    from ztforce.cache import fits_path, make_cache

    cache = make_cache(tmp_path / "new_cache")
    p = fits_path(cache, 1, 1, 1, "g", 1.0)
    assert p.parent.exists()


# ── PSF parsing ───────────────────────────────────────────────────────────────


def _make_synthetic_psf_file(path: Path, psf_size: int = 11, n_tables: int = 3) -> None:
    """Write a minimal valid DAOPhot PSF sidecar for testing."""
    x_cen, y_cen = 1535.5, 1539.5
    norm_factor = 1000.0
    sigma_x = sigma_y = 1.5
    header = (
        f" GAUSSIAN  {psf_size:3d}    2    {n_tables}    0   14.000"
        f"  {norm_factor:12.3f}  {x_cen}  {y_cen}\n"
    )
    sigmas = f"  {sigma_x:.6E} {sigma_y:.6E}\n"

    c = psf_size // 2
    row_idx, col_idx = np.mgrid[0:psf_size, 0:psf_size]
    gauss = norm_factor * np.exp(-0.5 * ((col_idx - c) ** 2 / sigma_x**2 + (row_idx - c) ** 2 / sigma_y**2))
    table0 = np.zeros((psf_size, psf_size))
    table1 = gauss * 0.01
    table2 = gauss * -0.01

    def _fmt_table(t):
        """Format a 2D array as fixed-width scientific notation rows."""
        lines = []
        flat = t.flatten()
        for i in range(0, len(flat), 6):
            chunk = flat[i : i + 6]
            lines.append("  " + " ".join(f"{v:.6E}" for v in chunk))
        return "\n".join(lines) + "\n"

    with open(path, "w") as f:
        f.write(header)
        f.write(sigmas)
        f.write(_fmt_table(table0))
        f.write(_fmt_table(table1))
        f.write(_fmt_table(table2))


def test_parse_daophot_psf(tmp_path):
    """parse_daophot_psf returns correct metadata and table shape."""
    from ztforce.psf import parse_daophot_psf

    psf_file = tmp_path / "test.psf"
    _make_synthetic_psf_file(psf_file, psf_size=11, n_tables=3)

    parsed = parse_daophot_psf(psf_file)
    assert parsed["psf_size"] == 11
    assert parsed["n_tables"] == 3
    assert parsed["tables"].shape == (3, 11, 11)
    assert parsed["norm_factor"] == 1000.0


def test_reconstruct_psf_normalized(tmp_path):
    """Reconstructed PSF sums to 1 and has no negative values."""
    from ztforce.psf import parse_daophot_psf, reconstruct_psf

    psf_file = tmp_path / "test.psf"
    _make_synthetic_psf_file(psf_file, psf_size=11, n_tables=3)
    parsed = parse_daophot_psf(psf_file)

    psf = reconstruct_psf(parsed, parsed["x_cen"], parsed["y_cen"])
    assert abs(psf.sum() - 1.0) < 1e-6
    assert psf.min() >= 0


def test_reconstruct_psf_peak_at_center(tmp_path):
    """For a symmetric PSF model, the peak is at the stamp center."""
    from ztforce.psf import parse_daophot_psf, reconstruct_psf

    psf_file = tmp_path / "test.psf"
    _make_synthetic_psf_file(psf_file, psf_size=11, n_tables=3)
    parsed = parse_daophot_psf(psf_file)

    psf = reconstruct_psf(parsed, parsed["x_cen"], parsed["y_cen"])
    c = 11 // 2
    peak_row, peak_col = np.unravel_index(psf.argmax(), psf.shape)
    assert abs(peak_row - c) <= 1
    assert abs(peak_col - c) <= 1


def test_parse_daophot_psf_bad_file(tmp_path):
    """PSFBuildError is raised for a malformed sidecar file."""
    from ztforce.exceptions import PSFBuildError
    from ztforce.psf import parse_daophot_psf

    bad = tmp_path / "bad.psf"
    bad.write_text("GARBAGE LINE\n0.0 0.0\n")
    with pytest.raises(PSFBuildError):
        parse_daophot_psf(bad)


# ── lightcurve ────────────────────────────────────────────────────────────────


def _make_lc(n_det=10, n_nondet=2, band="g"):
    """Helper: create a Lightcurve with n_det detections and n_nondet upper limits."""
    from ztforce.lightcurve import Lightcurve

    lc = Lightcurve(ra=150.0, dec=2.0)
    for i in range(n_det):
        lc.add_epoch(
            obsjd=2459000 + i * 5,
            band=band,
            flux=1000.0 + i * 50,
            flux_err=30.0,
            mag=18.8,
            mag_err=0.03,
            zero_point=26.3,
            flags=0,
            mag_limit=21.0,
        )
    for i in range(n_nondet):
        lc.add_epoch(
            obsjd=2459200 + i * 5,
            band=band,
            flux=-10.0,
            flux_err=30.0,
            mag=float("nan"),
            mag_err=float("nan"),
            zero_point=26.3,
            flags=0,
            mag_limit=21.5,
        )
    return lc


def test_lightcurve_len():
    """len(lc) equals total number of epochs including non-detections."""
    lc = _make_lc(10, 2)
    assert len(lc) == 12


def test_lightcurve_detections():
    """detection flag is True only for SNR >= 3 epochs with flags=0."""
    lc = _make_lc(10, 2)
    df = lc.df
    assert df["detection"].sum() == 10
    assert (~df["detection"]).sum() == 2


def test_lightcurve_stack_ivw():
    """stack() uses inverse-variance weighting: result = sum(f/σ²)/sum(1/σ²)."""
    from ztforce.lightcurve import Lightcurve

    lc = Lightcurve(ra=0.0, dec=0.0)
    for flux, ferr in [(1000.0, 10.0), (1200.0, 10.0)]:
        lc.add_epoch(2459000, "g", flux, ferr, 18.0, 0.01, 26.3, 0)
    s = lc.stack()
    # IVW: (1000/100 + 1200/100) / (1/100 + 1/100) = 1100
    assert abs(s.loc["g", "flux_stack"] - 1100.0) < 0.01


def test_lightcurve_save_load_roundtrip(tmp_path):
    """Lightcurve.save() / Lightcurve.load() preserve all epoch data exactly."""
    from ztforce.lightcurve import Lightcurve

    lc = _make_lc(n_det=5, n_nondet=1)
    path = tmp_path / "lc.ecsv"
    lc.save(path)
    lc2 = Lightcurve.load(path)
    assert len(lc2) == len(lc)
    assert lc2.ra == lc.ra
    assert lc2.dec == lc.dec
    np.testing.assert_allclose(lc2.df["flux"].values, lc.df["flux"].values)


def test_lightcurve_bands_order():
    """bands property returns bands in canonical g/r/i order regardless of insertion order."""
    from ztforce.lightcurve import Lightcurve

    lc = Lightcurve(ra=0.0, dec=0.0)
    for band in ["i", "g", "r"]:
        lc.add_epoch(2459000, band, 1000, 10, 18.0, 0.01, 26.3, 0)
    assert lc.bands == ["g", "r", "i"]


def test_lightcurve_repr():
    """repr includes RA, dec, and epoch count."""
    lc = _make_lc(3, 0)
    assert "150" in repr(lc)
    assert "n_epochs=3" in repr(lc)


# ── forced photometry integration ─────────────────────────────────────────────


def _make_synthetic_ztf_image(tmp_path, fwhm_px=3.0, n_sources=5):
    """Generate a synthetic ZTF-like FITS image with a known point source at center."""
    from astropy.io import fits
    from astropy.wcs import WCS

    size = 256
    rng = np.random.default_rng(0)
    data = rng.normal(0, 10, (size, size)).astype(np.float32)

    cy, cx = size // 2, size // 2
    y, x = np.mgrid[0:size, 0:size]
    sigma = fwhm_px / 2.355
    src = 5000.0 * np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / sigma**2)
    data += src.astype(np.float32)

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [cx + 1, cy + 1]
    wcs.wcs.cdelt = [-0.000281, 0.000281]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdr = wcs.to_header()
    hdr["MAGZP"] = 26.3
    hdr["OBSJD"] = 2459000.0
    hdr["GAIN"] = 6.2
    hdr["MEDFWHM"] = fwhm_px
    hdr["MAGLIM"] = 21.0
    hdr["RADESYS"] = "ICRS"

    fpath = tmp_path / "synth.fits"
    fits.writeto(str(fpath), data, hdr, overwrite=True)
    return fpath, cx, cy, size


def test_forced_phot_detects_injected_source(tmp_path):
    """Forced photometry recovers a bright injected source within 30% of true flux."""
    from ztforce.config import ZTForceConfig
    from ztforce.image import ZTFImage
    from ztforce.psf import forced_phot_at_position, parse_daophot_psf

    fits_fpath, cx, cy, size = _make_synthetic_ztf_image(tmp_path, fwhm_px=3.0)
    psf_fpath = tmp_path / "synth.psf"
    _make_synthetic_psf_file(psf_fpath, psf_size=11, n_tables=3)

    config = ZTForceConfig(irsa_user="u", irsa_pass="p")
    img = ZTFImage(str(fits_fpath), "g", config)
    parsed_psf = parse_daophot_psf(psf_fpath)
    parsed_psf["sigmas"] = [3.0 / 2.355, 3.0 / 2.355]

    coord = img.pixel_to_sky(float(cx), float(cy))
    result = forced_phot_at_position(img, parsed_psf, coord)

    assert result["flags"] == 0, f"Unexpected flags: {result['flags']}"
    assert result["flux"] > 0
    # Matched filter returns total integrated flux ≈ peak * 2π * σ²
    sigma = 3.0 / 2.355
    expected_flux = 5000.0 * 2 * np.pi * sigma**2
    assert (
        abs(result["flux"] - expected_flux) / expected_flux < 0.3
    ), f"Flux {result['flux']:.0f} too far from expected {expected_flux:.0f}"


def test_forced_phot_edge_returns_nan(tmp_path):
    """Position at the image edge returns flags=1 and NaN flux."""
    from ztforce.config import ZTForceConfig
    from ztforce.image import ZTFImage
    from ztforce.psf import forced_phot_at_position, parse_daophot_psf

    fits_fpath, cx, cy, size = _make_synthetic_ztf_image(tmp_path, fwhm_px=3.0)
    psf_fpath = tmp_path / "synth.psf"
    _make_synthetic_psf_file(psf_fpath, psf_size=11, n_tables=3)

    config = ZTForceConfig(irsa_user="u", irsa_pass="p")
    img = ZTFImage(str(fits_fpath), "g", config)
    parsed_psf = parse_daophot_psf(psf_fpath)

    edge_coord = img.pixel_to_sky(1.0, 1.0)
    result = forced_phot_at_position(img, parsed_psf, edge_coord)

    assert result["flags"] == 1
    assert np.isnan(result["flux"])
