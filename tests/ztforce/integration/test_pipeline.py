"""Tests for ztforce.pipeline (orchestration layer)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_metadata_row(field=468, ccdid=3, qid=2, obsjd=2459000.0):
    """Return a one-row metadata DataFrame as query_sci_metadata would."""
    return pd.DataFrame(
        [
            {
                "field": field,
                "ccdid": ccdid,
                "qid": qid,
                "obsjd": obsjd,
                "filtercode": "zg",
                "filefracday": "20210101001234",
                "paddedfield": f"{field:06d}",
            }
        ]
    )


def _write_synthetic_fits(path: Path, size: int = 64) -> None:
    """Write a minimal ZTF-like FITS file to disk."""
    cx, cy = size // 2, size // 2
    rng = np.random.default_rng(1)
    data = rng.normal(100.0, 10.0, (size, size)).astype(np.float32)
    sigma = 3.0 / 2.355
    y, x = np.mgrid[0:size, 0:size]
    data += (5000.0 * np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / sigma**2)).astype(np.float32)

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [cx + 1, cy + 1]
    wcs.wcs.cdelt = [-0.000281, 0.000281]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdr = wcs.to_header()
    hdr.update(MAGZP=26.3, OBSJD=2459000.0, GAIN=6.2, MEDFWHM=3.0, MAGLIM=21.0, RADESYS="ICRS")
    fits.writeto(str(path), data, hdr, overwrite=True)


def _write_synthetic_psf(path: Path) -> None:
    """Write a minimal DAOPhot PSF sidecar to disk."""
    psf_size = 11
    sigma = 1.5
    norm = 1000.0
    header = f" GAUSSIAN  {psf_size:3d}    2    3    0   14.000  {norm:12.3f}  1535.5  1539.5\n"
    sigmas_line = f"  {sigma:.6E} {sigma:.6E}\n"

    c = psf_size // 2
    row_idx, col_idx = np.mgrid[0:psf_size, 0:psf_size]
    gauss = norm * np.exp(-0.5 * ((col_idx - c) ** 2 + (row_idx - c) ** 2) / sigma**2)
    t1 = (col_idx - c) / (c + 1) * gauss * 0.2
    t2 = (row_idx - c) / (c + 1) * gauss * 0.2
    tables = [np.zeros((psf_size, psf_size)), t1, t2]

    def _fmt(t):
        flat = t.flatten()
        rows = []
        for i in range(0, len(flat), 6):
            rows.append("  " + " ".join(f"{v:.6E}" for v in flat[i : i + 6]))
        return "\n".join(rows) + "\n"

    with open(path, "w") as f:
        f.write(header)
        f.write(sigmas_line)
        for t in tables:
            f.write(_fmt(t))


# ── _worker_kwargs ────────────────────────────────────────────────────────────


def test_worker_kwargs_contains_all_fields(mock_config):
    """_worker_kwargs extracts all picklable config fields."""
    from ztforce.pipeline import _worker_kwargs

    kwargs = _worker_kwargs(mock_config)
    for key in ("irsa_user", "irsa_pass", "default_gain"):
        assert key in kwargs


def test_worker_kwargs_values_match_config(mock_config):
    """_worker_kwargs values match the config fields."""
    from ztforce.pipeline import _worker_kwargs

    kwargs = _worker_kwargs(mock_config)
    assert kwargs["irsa_user"] == mock_config.irsa_user
    assert kwargs["default_gain"] == mock_config.default_gain


# ── _download_all ────────────────────────────────────────────────────────────


def test_download_all_returns_sorted_triples(tmp_path, mock_config):
    """_download_all calls download functions and returns (row, fits_path, psf_path) sorted by obsjd."""
    from ztforce.pipeline import _download_all

    fits_path = tmp_path / "img.fits"
    psf_fpath = tmp_path / "img.psf"
    _write_synthetic_fits(fits_path)
    _write_synthetic_psf(psf_fpath)

    df = pd.concat(
        [_make_metadata_row(obsjd=2459002.0), _make_metadata_row(obsjd=2459001.0)], ignore_index=True
    )

    with (
        mock.patch("ztforce.pipeline.download_fits", return_value=fits_path),
        mock.patch("ztforce.pipeline.download_psf_sidecar", return_value=psf_fpath),
        mock.patch("ztforce.pipeline.build_sci_url", return_value="http://fake/url"),
    ):
        results = _download_all(df, 150.0, 2.0, "g", tmp_path, mock_config, n_workers=1)

    assert len(results) == 2
    obsjds = [float(r[0]["obsjd"]) for r in results]
    assert obsjds == sorted(obsjds)


def test_download_all_skips_failed_downloads(tmp_path, mock_config):
    """_download_all silently drops images whose download raises."""
    from ztforce.pipeline import _download_all

    df = _make_metadata_row()

    with (
        mock.patch("ztforce.pipeline.download_fits", side_effect=Exception("network error")),
        mock.patch("ztforce.pipeline.build_sci_url", return_value="http://fake/url"),
    ):
        results = _download_all(df, 150.0, 2.0, "g", tmp_path, mock_config, n_workers=1)

    assert results == []


# ── _run_psf_parallel ─────────────────────────────────────────────────────────


def test_run_psf_parallel_calls_worker(tmp_path, mock_config):
    """_run_psf_parallel submits one job per image and returns results sorted by obsjd."""
    from concurrent.futures import ThreadPoolExecutor

    from ztforce.pipeline import _run_psf_parallel
    from ztforce.utils import flux_to_ab_mag

    fits_path = tmp_path / "img.fits"
    psf_fpath = tmp_path / "img.psf"
    _write_synthetic_fits(fits_path)
    _write_synthetic_psf(psf_fpath)

    df = _make_metadata_row()
    image_triples = [(df.iloc[0], fits_path, psf_fpath)]

    mag, merr = flux_to_ab_mag(1000.0, 26.3, 50.0)
    fake_result = dict(
        flux=1000.0,
        flux_err=50.0,
        mag=mag,
        mag_err=merr,
        flags=0,
        x_fit=32.0,
        y_fit=32.0,
        obsjd=2459000.0,
        zero_point=26.3,
        mag_limit=21.0,
        image_id="test",
        band="g",
    )

    # Replace ProcessPoolExecutor with ThreadPoolExecutor so mock survives pickling
    with (
        mock.patch("ztforce.pipeline.ProcessPoolExecutor", ThreadPoolExecutor),
        mock.patch("ztforce.pipeline._process_one_image", return_value=fake_result),
    ):
        results = _run_psf_parallel(image_triples, 150.0, 2.0, "g", mock_config, n_workers=1)

    assert len(results) == 1
    assert results[0]["flux"] == pytest.approx(1000.0)


# ── run_forced_photometry (empty downloads) ───────────────────────────────────


def test_pipeline_empty_downloads_skips_band(tmp_path, mock_config):
    """run_forced_photometry skips a band when all downloads fail."""
    from ztforce.pipeline import run_forced_photometry

    df = _make_metadata_row()
    with (
        mock.patch("ztforce.pipeline.query_sci_metadata", return_value=df),
        mock.patch("ztforce.pipeline._download_all", return_value=[]),
    ):
        result = run_forced_photometry(
            150.0, 2.0, bands=["g"], data_dir=tmp_path / "cache", config=mock_config
        )

    assert result == {}


# ── _process_one_image ────────────────────────────────────────────────────────


def test_process_one_image_returns_dict(tmp_path, mock_config):
    """_process_one_image returns a result dict with expected keys."""
    from ztforce.pipeline import _process_one_image, _worker_kwargs

    fits_path = tmp_path / "img.fits"
    psf_path = tmp_path / "img.psf"
    _write_synthetic_fits(fits_path)
    _write_synthetic_psf(psf_path)

    kwargs = _worker_kwargs(mock_config)
    result = _process_one_image(str(fits_path), str(psf_path), 150.0, 2.0, "g", "test-id", **kwargs)

    for key in ("flux", "flux_err", "mag", "flags", "obsjd", "zero_point", "band"):
        assert key in result, f"Missing key: {key}"


def test_process_one_image_bad_fits_returns_flags2(tmp_path, mock_config):
    """_process_one_image returns flags=2 when the FITS file is corrupt."""
    from ztforce.pipeline import _process_one_image, _worker_kwargs

    (tmp_path / "bad.fits").write_text("GARBAGE")
    psf_path = tmp_path / "img.psf"
    _write_synthetic_psf(psf_path)

    kwargs = _worker_kwargs(mock_config)
    result = _process_one_image(
        str(tmp_path / "bad.fits"), str(psf_path), 150.0, 2.0, "g", "err-id", **kwargs
    )
    assert result["flags"] == 2


# ── run_forced_photometry (cache hit) ─────────────────────────────────────────


def test_cache_hit_skips_all_computation(tmp_path, mock_config):
    """run_forced_photometry loads from cache and never calls query_sci_metadata."""
    from ztforce.cache import lightcurve_path, make_cache
    from ztforce.lightcurve import Lightcurve
    from ztforce.pipeline import _cache_key, run_forced_photometry
    from ztforce.utils import flux_to_ab_mag

    cache = make_cache(tmp_path / "cache")
    lc_path = lightcurve_path(cache, 150.0, 2.0, "g")
    lc_path.parent.mkdir(parents=True, exist_ok=True)

    lc_pre = Lightcurve(ra=150.0, dec=2.0)
    lc_pre.cache_key = _cache_key(mock_config, None)
    mag, merr = flux_to_ab_mag(1000.0, 26.3, 50.0)
    lc_pre.add_epoch(2459000.0, "g", 1000.0, 50.0, mag, merr, 26.3, 0)
    lc_pre.save(lc_path)

    with mock.patch("ztforce.pipeline.query_sci_metadata") as mock_query:
        result = run_forced_photometry(
            150.0, 2.0, bands=["g"], data_dir=tmp_path / "cache", config=mock_config
        )

    mock_query.assert_not_called()
    assert "g" in result
    assert len(result["g"]) == 1


def test_cache_hit_returns_correct_lightcurve(tmp_path, mock_config):
    """The cached lightcurve has the correct ra/dec metadata after reload."""
    from ztforce.cache import lightcurve_path, make_cache
    from ztforce.lightcurve import Lightcurve
    from ztforce.pipeline import _cache_key, run_forced_photometry
    from ztforce.utils import flux_to_ab_mag

    cache = make_cache(tmp_path / "cache")
    lc_path = lightcurve_path(cache, 150.0, 2.0, "g")
    lc_path.parent.mkdir(parents=True, exist_ok=True)

    lc_pre = Lightcurve(ra=150.0, dec=2.0)
    lc_pre.cache_key = _cache_key(mock_config, None)
    mag, merr = flux_to_ab_mag(500.0, 26.3, 25.0)
    lc_pre.add_epoch(2459000.0, "g", 500.0, 25.0, mag, merr, 26.3, 0)
    lc_pre.save(lc_path)

    result = run_forced_photometry(150.0, 2.0, bands=["g"], data_dir=tmp_path / "cache", config=mock_config)
    assert result["g"].ra == pytest.approx(150.0)
    assert result["g"].dec == pytest.approx(2.0)


# ── run_forced_photometry (no images) ─────────────────────────────────────────


def test_cache_hit_stale_key_triggers_recompute(tmp_path, mock_config):
    """A cached lightcurve with a mismatched cache_key is recomputed, not returned."""
    from ztforce.cache import lightcurve_path, make_cache
    from ztforce.exceptions import NoImagesFoundError
    from ztforce.lightcurve import Lightcurve
    from ztforce.pipeline import run_forced_photometry
    from ztforce.utils import flux_to_ab_mag

    cache = make_cache(tmp_path / "cache")
    lc_path = lightcurve_path(cache, 150.0, 2.0, "g")
    lc_path.parent.mkdir(parents=True, exist_ok=True)

    lc_pre = Lightcurve(ra=150.0, dec=2.0)
    lc_pre.cache_key = "stale_key_000"  # won't match current config hash
    mag, merr = flux_to_ab_mag(1000.0, 26.3, 50.0)
    lc_pre.add_epoch(2459000.0, "g", 1000.0, 50.0, mag, merr, 26.3, 0)
    lc_pre.save(lc_path)

    with mock.patch("ztforce.pipeline.query_sci_metadata") as mock_query:
        mock_query.side_effect = NoImagesFoundError("none")
        run_forced_photometry(150.0, 2.0, bands=["g"], data_dir=tmp_path / "cache", config=mock_config)

    mock_query.assert_called_once()


def test_force_recompute_ignores_cache(tmp_path, mock_config):
    """force_recompute=True bypasses an existing cached lightcurve."""
    from ztforce.cache import lightcurve_path, make_cache
    from ztforce.exceptions import NoImagesFoundError
    from ztforce.lightcurve import Lightcurve
    from ztforce.pipeline import run_forced_photometry
    from ztforce.utils import flux_to_ab_mag

    cache = make_cache(tmp_path / "cache")
    lc_path = lightcurve_path(cache, 150.0, 2.0, "g")
    lc_path.parent.mkdir(parents=True, exist_ok=True)
    lc_pre = Lightcurve(ra=150.0, dec=2.0)
    mag, merr = flux_to_ab_mag(1000.0, 26.3, 50.0)
    lc_pre.add_epoch(2459000.0, "g", 1000.0, 50.0, mag, merr, 26.3, 0)
    lc_pre.save(lc_path)

    with mock.patch("ztforce.pipeline.query_sci_metadata") as mock_query:
        mock_query.side_effect = NoImagesFoundError("none")
        result = run_forced_photometry(
            150.0,
            2.0,
            bands=["g"],
            data_dir=tmp_path / "cache",
            config=mock_config,
            force_recompute=True,
        )

    mock_query.assert_called_once()
    assert result == {}


def test_no_images_returns_empty_dict(tmp_path, mock_config):
    """run_forced_photometry returns {} when query finds no images."""
    from ztforce.exceptions import NoImagesFoundError
    from ztforce.pipeline import run_forced_photometry

    with mock.patch("ztforce.pipeline.query_sci_metadata", side_effect=NoImagesFoundError("none")):
        result = run_forced_photometry(
            150.0, 2.0, bands=["g"], data_dir=tmp_path / "cache", config=mock_config
        )

    assert result == {}


# ── run_forced_photometry (mocked download + photometry) ─────────────────────


def test_pipeline_full_mocked(tmp_path, mock_config):
    """run_forced_photometry assembles a Lightcurve from mocked download + worker results."""
    from ztforce.pipeline import run_forced_photometry
    from ztforce.utils import flux_to_ab_mag

    fits_path = tmp_path / "img.fits"
    psf_fpath = tmp_path / "img.psf"
    _write_synthetic_fits(fits_path)
    _write_synthetic_psf(psf_fpath)

    df = _make_metadata_row()
    mag, merr = flux_to_ab_mag(1000.0, 26.3, 50.0)
    fake_result = dict(
        flux=1000.0,
        flux_err=50.0,
        mag=mag,
        mag_err=merr,
        flags=0,
        x_fit=32.0,
        y_fit=32.0,
        obsjd=2459000.0,
        zero_point=26.3,
        mag_limit=21.0,
        image_id="468-3-2-2459000.000",
        band="g",
    )

    with (
        mock.patch("ztforce.pipeline.query_sci_metadata", return_value=df),
        mock.patch("ztforce.pipeline._download_all", return_value=[(df.iloc[0], fits_path, psf_fpath)]),
        mock.patch("ztforce.pipeline._run_psf_parallel", return_value=[fake_result]),
    ):
        result = run_forced_photometry(
            150.0, 2.0, bands=["g"], data_dir=tmp_path / "cache", config=mock_config
        )

    assert "g" in result
    lc = result["g"]
    assert len(lc) == 1
    assert lc.df.iloc[0]["flux"] == pytest.approx(1000.0)
    assert lc.df.iloc[0]["detection"]


def test_pipeline_saves_lightcurve_to_cache(tmp_path, mock_config):
    """run_forced_photometry saves the result ECSV to the expected cache path."""
    from ztforce.cache import lightcurve_path, make_cache
    from ztforce.pipeline import run_forced_photometry
    from ztforce.utils import flux_to_ab_mag

    fits_path = tmp_path / "img.fits"
    psf_fpath = tmp_path / "img.psf"
    _write_synthetic_fits(fits_path)
    _write_synthetic_psf(psf_fpath)

    df = _make_metadata_row()
    mag, merr = flux_to_ab_mag(1000.0, 26.3, 50.0)
    fake_result = dict(
        flux=1000.0,
        flux_err=50.0,
        mag=mag,
        mag_err=merr,
        flags=0,
        x_fit=32.0,
        y_fit=32.0,
        obsjd=2459000.0,
        zero_point=26.3,
        mag_limit=21.0,
        image_id="468-3-2-2459000.000",
        band="g",
    )

    with (
        mock.patch("ztforce.pipeline.query_sci_metadata", return_value=df),
        mock.patch("ztforce.pipeline._download_all", return_value=[(df.iloc[0], fits_path, psf_fpath)]),
        mock.patch("ztforce.pipeline._run_psf_parallel", return_value=[fake_result]),
    ):
        run_forced_photometry(150.0, 2.0, bands=["g"], data_dir=tmp_path / "cache", config=mock_config)

    expected = lightcurve_path(make_cache(tmp_path / "cache"), 150.0, 2.0, "g")
    assert expected.exists()


# ── run_forced_photometry_batch ───────────────────────────────────────────────


def test_batch_delegates_to_run_per_target(mock_config):
    """run_forced_photometry_batch calls run_forced_photometry once per target."""
    from ztforce.pipeline import run_forced_photometry_batch

    targets = [
        SkyCoord(ra=150.0, dec=2.0, unit="deg"),
        SkyCoord(ra=200.0, dec=-5.0, unit="deg"),
    ]

    with mock.patch("ztforce.pipeline.run_forced_photometry", return_value={}) as mock_rfp:
        result = run_forced_photometry_batch(targets, bands=["g"], config=mock_config)

    assert mock_rfp.call_count == 2
    assert len(result) == 2


def test_batch_passes_ra_dec_correctly(mock_config):
    """run_forced_photometry_batch passes the correct ra/dec for each target."""
    from ztforce.pipeline import run_forced_photometry_batch

    targets = [SkyCoord(ra=123.456, dec=-7.89, unit="deg")]

    calls = []

    def _capture(*args, **kwargs):
        calls.append(
            (
                kwargs.get("ra", args[0] if args else None),
                kwargs.get("dec", args[1] if len(args) > 1 else None),
            )
        )
        return {}

    with mock.patch("ztforce.pipeline.run_forced_photometry", side_effect=_capture):
        run_forced_photometry_batch(targets, bands=["g"], config=mock_config)

    assert calls[0][0] == pytest.approx(123.456, rel=1e-5)
    assert calls[0][1] == pytest.approx(-7.89, rel=1e-5)
