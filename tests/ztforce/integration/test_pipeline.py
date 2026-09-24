"""Tests for ztforce.pipeline (orchestration layer)."""

from __future__ import annotations

import warnings
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
    """Return a one-row metadata DataFrame, like one band of query_sci_metadata_bands."""
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


# ── _download_epoch ───────────────────────────────────────────────────────────


def test_download_epoch_returns_triple(tmp_path, mock_config):
    """_download_epoch returns (row, fits_path, psf_path) on success."""
    from ztforce.pipeline import _download_epoch

    row = _make_metadata_row().iloc[0]
    with (
        mock.patch("ztforce.pipeline.build_sci_url", return_value="http://fake/url"),
        mock.patch("ztforce.pipeline.download_fits") as mock_fits,
        mock.patch("ztforce.pipeline.download_psf_sidecar") as mock_psf,
    ):
        result_row, fits_p, psf_p = _download_epoch(row, tmp_path, 150.0, 2.0, mock_config)

    assert result_row["field"] == row["field"]
    assert fits_p.suffix == ".fits"
    assert psf_p.suffix == ".psf"
    mock_fits.assert_called_once()
    mock_psf.assert_called_once()


def test_download_epoch_propagates_exception(tmp_path, mock_config):
    """_download_epoch re-raises when download_fits fails."""
    from ztforce.pipeline import _download_epoch

    row = _make_metadata_row().iloc[0]
    with (
        mock.patch("ztforce.pipeline.build_sci_url", return_value="http://fake/url"),
        mock.patch("ztforce.pipeline.download_fits", side_effect=Exception("timeout")),
    ):
        with pytest.raises(Exception, match="timeout"):
            _download_epoch(row, tmp_path, 150.0, 2.0, mock_config)


# ── _process_one_epoch ────────────────────────────────────────────────────────


def test_process_one_epoch_returns_dict(tmp_path, mock_config):
    """_process_one_epoch returns a result dict with expected keys."""
    from ztforce.pipeline import _process_one_epoch

    fits_path = tmp_path / "img.fits"
    psf_path = tmp_path / "img.psf"
    _write_synthetic_fits(fits_path)
    _write_synthetic_psf(psf_path)

    result = _process_one_epoch(str(fits_path), str(psf_path), 150.0, 2.0, "g", "test-id", mock_config)

    for key in ("flux", "flux_err", "mag", "flags", "obsjd", "zero_point", "band"):
        assert key in result, f"Missing key: {key}"


def test_process_one_epoch_bad_fits_returns_flags2(tmp_path, mock_config):
    """_process_one_epoch returns flags=2 when the FITS file is corrupt."""
    from ztforce.pipeline import _process_one_epoch

    (tmp_path / "bad.fits").write_text("GARBAGE")
    psf_path = tmp_path / "img.psf"
    _write_synthetic_psf(psf_path)

    result = _process_one_epoch(
        str(tmp_path / "bad.fits"), str(psf_path), 150.0, 2.0, "g", "err-id", mock_config
    )
    assert result["flags"] == 2


# ── quality flags / uncertainty rescaling ─────────────────────────────────────


@pytest.mark.parametrize(
    ("infobits", "scisigpix", "seeing", "expected"),
    [
        (0, 10.0, 2.0, 0),
        (2**25, 10.0, 2.0, 4),
        (2**25 + 1, 10.0, 2.0, 4),
        (2**25 - 1, 10.0, 2.0, 0),  # lower bits alone are not fatal
        (0, 30.0, 2.0, 8),
        (0, 10.0, 4.5, 16),
        (2**26, 30.0, 4.5, 28),
    ],
)
def test_quality_flags(infobits, scisigpix, seeing, expected):
    """ZFPS section 6.1 cuts map onto their flag bits."""
    from ztforce.pipeline import _quality_flags

    assert _quality_flags(dict(infobits=infobits, scisigpix=scisigpix, seeing=seeing)) == expected


def test_process_one_epoch_records_quality_metrics(tmp_path, mock_config):
    """_process_one_epoch stores chisq, infobits, seeing and scisigpix, and flags bad calibration."""
    from ztforce.pipeline import _process_one_epoch

    fits_path = tmp_path / "img.fits"
    psf_path = tmp_path / "img.psf"
    _write_synthetic_fits(fits_path)
    with fits.open(fits_path, mode="update") as hdul:
        hdul[0].header.update(INFOBITS=2**25, SEEING=2.0, PIXSCALE=1.01)
    _write_synthetic_psf(psf_path)

    result = _process_one_epoch(str(fits_path), str(psf_path), 150.0, 2.0, "g", "id", mock_config)

    assert np.isfinite(result["chisq"]) and result["chisq"] > 0
    assert result["infobits"] == 2**25
    assert result["seeing"] == pytest.approx(2.02)
    assert result["scisigpix"] == pytest.approx(10.0, rel=0.2)  # synthetic sky sigma is 10 DN
    assert result["flags"] & 4


# ── run_forced_photometry (empty downloads) ───────────────────────────────────


def test_pipeline_failed_downloads_kept_as_flagged_rows(tmp_path, mock_config):
    """A band whose downloads all fail is still returned, as flagged NaN rows, with a warning."""
    from ztforce.pipeline import run_forced_photometry

    df = _make_metadata_row()
    with (
        mock.patch("ztforce.pipeline.query_sci_metadata_bands", return_value={"g": df}),
        mock.patch("ztforce.pipeline.download_fits", side_effect=Exception("network error")),
        mock.patch("ztforce.pipeline.build_sci_url", return_value="http://fake/url"),
        pytest.warns(UserWarning, match="1 failed"),
    ):
        result = run_forced_photometry(
            150.0, 2.0, bands=["g"], data_dir=tmp_path / "cache", config=mock_config, show_progress=False
        )

    row = result["g"].df.iloc[0]
    assert row["flags"] & 64
    assert np.isnan(row["flux"])


# ── run_forced_photometry (cache hit) ─────────────────────────────────────────


def test_cache_hit_skips_all_computation(tmp_path, mock_config):
    """run_forced_photometry loads from cache and never queries metadata."""
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

    with mock.patch("ztforce.pipeline.query_sci_metadata_bands") as mock_query:
        result = run_forced_photometry(
            150.0, 2.0, bands=["g"], data_dir=tmp_path / "cache", config=mock_config, show_progress=False
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

    result = run_forced_photometry(
        150.0, 2.0, bands=["g"], data_dir=tmp_path / "cache", config=mock_config, show_progress=False
    )
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
    lc_pre.cache_key = "stale_key_000"
    mag, merr = flux_to_ab_mag(1000.0, 26.3, 50.0)
    lc_pre.add_epoch(2459000.0, "g", 1000.0, 50.0, mag, merr, 26.3, 0)
    lc_pre.save(lc_path)

    with mock.patch("ztforce.pipeline.query_sci_metadata_bands") as mock_query:
        mock_query.side_effect = NoImagesFoundError("none")
        run_forced_photometry(
            150.0, 2.0, bands=["g"], data_dir=tmp_path / "cache", config=mock_config, show_progress=False
        )

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

    with mock.patch("ztforce.pipeline.query_sci_metadata_bands") as mock_query:
        mock_query.side_effect = NoImagesFoundError("none")
        result = run_forced_photometry(
            150.0,
            2.0,
            bands=["g"],
            data_dir=tmp_path / "cache",
            config=mock_config,
            force_recompute=True,
            show_progress=False,
        )

    mock_query.assert_called_once()
    assert result == {}


def test_no_images_returns_empty_dict(tmp_path, mock_config):
    """run_forced_photometry returns {} when query finds no images."""
    from ztforce.exceptions import NoImagesFoundError
    from ztforce.pipeline import run_forced_photometry

    with mock.patch("ztforce.pipeline.query_sci_metadata_bands", side_effect=NoImagesFoundError("none")):
        result = run_forced_photometry(
            150.0, 2.0, bands=["g"], data_dir=tmp_path / "cache", config=mock_config, show_progress=False
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
        mock.patch("ztforce.pipeline.query_sci_metadata_bands", return_value={"g": df}),
        mock.patch("ztforce.pipeline.download_fits", return_value=fits_path),
        mock.patch("ztforce.pipeline.download_psf_sidecar", return_value=psf_fpath),
        mock.patch("ztforce.pipeline.build_sci_url", return_value="http://fake/url"),
        mock.patch("ztforce.pipeline._process_one_epoch", return_value=fake_result),
    ):
        result = run_forced_photometry(
            150.0, 2.0, bands=["g"], data_dir=tmp_path / "cache", config=mock_config, show_progress=False
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
        mock.patch("ztforce.pipeline.query_sci_metadata_bands", return_value={"g": df}),
        mock.patch("ztforce.pipeline.download_fits", return_value=fits_path),
        mock.patch("ztforce.pipeline.download_psf_sidecar", return_value=psf_fpath),
        mock.patch("ztforce.pipeline.build_sci_url", return_value="http://fake/url"),
        mock.patch("ztforce.pipeline._process_one_epoch", return_value=fake_result),
    ):
        run_forced_photometry(
            150.0, 2.0, bands=["g"], data_dir=tmp_path / "cache", config=mock_config, show_progress=False
        )

    expected = lightcurve_path(make_cache(tmp_path / "cache"), 150.0, 2.0, "g")
    assert expected.exists()


# ── run_forced_photometry (multiple bands) ────────────────────────────────────


def _fake_result(band: str, obsjd: float = 2459000.0) -> dict:
    from ztforce.utils import flux_to_ab_mag

    mag, merr = flux_to_ab_mag(1000.0, 26.3, 50.0)
    return dict(
        flux=1000.0,
        flux_err=50.0,
        mag=mag,
        mag_err=merr,
        flags=0,
        x_fit=32.0,
        y_fit=32.0,
        obsjd=obsjd,
        zero_point=26.3,
        mag_limit=21.0,
        image_id=f"468-3-2-{obsjd:.3f}",
        band=band,
    )


def test_metadata_queried_once_for_all_uncached_bands(tmp_path, mock_config):
    """Bands missing from the cache share one metadata query; cached bands are not re-queried."""
    from ztforce.cache import lightcurve_path, make_cache
    from ztforce.lightcurve import Lightcurve
    from ztforce.pipeline import _cache_key, run_forced_photometry

    cache = make_cache(tmp_path / "cache")
    lc_pre = Lightcurve(ra=150.0, dec=2.0)
    lc_pre.cache_key = _cache_key(mock_config, None)
    lc_pre.add_epoch(2459000.0, "g", 1000.0, 50.0, 18.8, 0.05, 26.3, 0)
    lc_pre.save(lightcurve_path(cache, 150.0, 2.0, "g"))

    metadata = {"r": _make_metadata_row(obsjd=2459001.0), "i": _make_metadata_row(obsjd=2459002.0)}
    with (
        mock.patch("ztforce.pipeline.query_sci_metadata_bands", return_value=metadata) as mock_query,
        mock.patch("ztforce.pipeline.download_fits"),
        mock.patch("ztforce.pipeline.download_psf_sidecar"),
        mock.patch("ztforce.pipeline.build_sci_url", return_value="http://fake/url"),
        mock.patch(
            "ztforce.pipeline._process_one_epoch", side_effect=lambda *a, **k: _fake_result(a[4], 2459001.0)
        ),
    ):
        result = run_forced_photometry(
            150.0,
            2.0,
            bands=["g", "r", "i"],
            data_dir=tmp_path / "cache",
            config=mock_config,
            show_progress=False,
        )

    mock_query.assert_called_once()
    assert mock_query.call_args.args[2] == ["r", "i"]
    assert set(result) == {"g", "r", "i"}
    for band in ("r", "i"):
        assert lightcurve_path(cache, 150.0, 2.0, band).exists()


def test_all_bands_submitted_before_first_fit(tmp_path, mock_config):
    """Every band's downloads are queued before any band is fitted, so the pool never idles."""
    from concurrent.futures import Future

    from ztforce.pipeline import run_forced_photometry

    events: list[tuple[str, str]] = []

    class _SyncExecutor:
        def submit(self, fn, *args, **kwargs):
            events.append(("submit", args[0]["filtercode"]))
            fut: Future = Future()
            fut.set_result(fn(*args, **kwargs))
            return fut

    def _fit(*args, **kwargs):
        events.append(("fit", args[4]))
        return _fake_result(args[4])

    df_g = _make_metadata_row(obsjd=2459000.0)
    df_r = _make_metadata_row(obsjd=2459001.0).assign(filtercode="zr")
    with (
        mock.patch("ztforce.pipeline.query_sci_metadata_bands", return_value={"g": df_g, "r": df_r}),
        mock.patch("ztforce.pipeline.download_fits"),
        mock.patch("ztforce.pipeline.download_psf_sidecar"),
        mock.patch("ztforce.pipeline.build_sci_url", return_value="http://fake/url"),
        mock.patch("ztforce.pipeline._process_one_epoch", side_effect=_fit),
    ):
        result = run_forced_photometry(
            150.0,
            2.0,
            bands=["g", "r"],
            data_dir=tmp_path / "cache",
            config=mock_config,
            show_progress=False,
            _download_executor=_SyncExecutor(),
        )

    assert events == [("submit", "zg"), ("submit", "zr"), ("fit", "g"), ("fit", "r")]
    assert set(result) == {"g", "r"}


# ── metadata quality cuts (skip / measure modes) ─────────────────────────────


def _meta(obsjd: float, infobits: int = 0, seeing: float = 2.0) -> pd.DataFrame:
    return _make_metadata_row(obsjd=obsjd).assign(infobits=infobits, seeing=seeing)


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (dict(infobits=0, seeing=2.0), 0),
        (dict(infobits=2**25, seeing=2.0), 4),
        (dict(infobits=2**26, seeing=2.0), 4),  # newer bad-calibration bit
        (dict(infobits=2**25 - 1, seeing=2.0), 0),
        (dict(infobits=0, seeing=4.5), 16),
        (dict(infobits=2**25, seeing=4.5), 20),
        (dict(), 0),  # metadata without the columns: nothing to decide
    ],
)
def test_metadata_flags(row, expected):
    """Metadata infobits >= 2**25 flags bad calibration; metadata seeing > 4 arcsec flags seeing."""
    from ztforce.pipeline import _metadata_flags

    assert _metadata_flags(pd.Series(row, dtype=object)) == expected


def _run_with_metadata(tmp_path, mock_config, metadata, measure_flagged, header_infobits=0):
    """Run the pipeline on mocked metadata; the fit reports a clean header (no bit 25)."""
    from ztforce.pipeline import run_forced_photometry

    def _fit(*args, **kwargs):
        res = _fake_result(args[4], obsjd=2459000.0)
        res["image_id"] = args[5]
        res["obsjd"] = float(args[5].rsplit("-", 1)[1])
        res["infobits"] = header_infobits
        return res

    with (
        mock.patch("ztforce.pipeline.query_sci_metadata_bands", return_value={"g": metadata}),
        mock.patch("ztforce.pipeline.download_fits") as mock_dl,
        mock.patch("ztforce.pipeline.download_psf_sidecar"),
        mock.patch("ztforce.pipeline.build_sci_url", return_value="http://fake/url"),
        mock.patch("ztforce.pipeline._process_one_epoch", side_effect=_fit),
    ):
        result = run_forced_photometry(
            150.0,
            2.0,
            bands=["g"],
            data_dir=tmp_path / "cache",
            config=mock_config,
            show_progress=False,
            measure_flagged=measure_flagged,
        )
    return result, mock_dl


def test_skip_mode_does_not_download_flagged_epoch(tmp_path, mock_config):
    """By default a metadata-flagged epoch is not downloaded but stays as a flagged NaN row."""
    metadata = pd.concat([_meta(2459000.0), _meta(2459001.0, infobits=2**25)], ignore_index=True)
    result, mock_dl = _run_with_metadata(tmp_path, mock_config, metadata, measure_flagged=False)

    assert mock_dl.call_count == 1  # only the good epoch
    df = result["g"].df
    assert len(df) == 2
    bad = df[df["obsjd"] == 2459001.0].iloc[0]
    assert bad["flags"] & 4
    assert (bad["field"], bad["ccdid"], bad["qid"]) == (468, 3, 2)  # unmeasured rows too
    assert np.isnan(bad["flux"])
    assert bad["infobits"] == 2**25
    assert not bad["detection"]
    assert df[df["obsjd"] == 2459000.0].iloc[0]["flags"] == 0


def test_measure_mode_flags_from_metadata_not_header(tmp_path, mock_config):
    """measure_flagged=True fits the epoch and flags it from metadata, though the header is clean."""
    result, mock_dl = _run_with_metadata(
        tmp_path, mock_config, _meta(2459001.0, infobits=2**25), measure_flagged=True, header_infobits=0
    )

    assert mock_dl.call_count == 1
    row = result["g"].df.iloc[0]
    assert row["flags"] & 4
    assert row["flux"] == pytest.approx(1000.0)
    assert row["infobits"] == 2**25  # the metadata value, not the header's 0
    assert not row["detection"]


def test_band_of_only_flagged_epochs_is_still_returned(tmp_path, mock_config):
    """A band whose every epoch is cut from metadata still gets a lightcurve of flagged rows."""
    result, mock_dl = _run_with_metadata(
        tmp_path, mock_config, _meta(2459001.0, seeing=5.0), measure_flagged=False
    )

    mock_dl.assert_not_called()
    df = result["g"].df
    assert len(df) == 1 and df.iloc[0]["flags"] & 16


def test_measure_flagged_is_part_of_the_cache_key(mock_config):
    """The two modes never share a cached lightcurve."""
    from ztforce.pipeline import _cache_key

    assert _cache_key(mock_config, None) == _cache_key(mock_config, None, False)
    assert _cache_key(mock_config, None, False) != _cache_key(mock_config, None, True)


# ── unavailable / failed epochs and retries ──────────────────────────────────


def _run_downloads(tmp_path, mock_config, metadata, download_side_effect, **kwargs):
    """Run the pipeline with a controllable download_fits; returns (result, download mock)."""
    from ztforce.pipeline import run_forced_photometry

    def _fit(*args, **kw):
        res = _fake_result(args[4])
        res["image_id"] = args[5]
        res["obsjd"] = float(args[5].rsplit("-", 1)[1])
        return res

    with (
        mock.patch("ztforce.pipeline.query_sci_metadata_bands", return_value={"g": metadata}),
        mock.patch("ztforce.pipeline.download_fits", side_effect=download_side_effect) as mock_dl,
        mock.patch("ztforce.pipeline.download_psf_sidecar"),
        mock.patch("ztforce.pipeline.build_sci_url", side_effect=lambda row, *a, **k: str(row["obsjd"])),
        mock.patch("ztforce.pipeline._process_one_epoch", side_effect=_fit),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        result = run_forced_photometry(
            150.0,
            2.0,
            bands=["g"],
            data_dir=tmp_path / "cache",
            config=mock_config,
            show_progress=False,
            **kwargs,
        )
    return result, mock_dl


def _three_epochs():
    return pd.concat([_meta(2459000.0), _meta(2459001.0), _meta(2459002.0)], ignore_index=True)


def _downloads(unavailable=(), failing=()):
    """download_fits side effect: 404 for `unavailable` obsjds, transient error for `failing`."""
    from ztforce.exceptions import ProductUnavailableError

    def _dl(url, dest, config):
        jd = float(url)
        if jd in unavailable:
            raise ProductUnavailableError(url, 404)
        if jd in failing:
            raise RuntimeError("timeout")
        return dest

    return _dl


def test_unavailable_and_failed_epochs_are_flagged_rows(tmp_path, mock_config):
    """404s get FLAG_UNAVAILABLE, other failures FLAG_DOWNLOAD_FAILED; no epoch disappears."""
    result, _ = _run_downloads(
        tmp_path, mock_config, _three_epochs(), _downloads(unavailable={2459001.0}, failing={2459002.0})
    )
    flags = result["g"].df.set_index("obsjd")["flags"]
    assert flags[2459000.0] == 0
    assert flags[2459001.0] & 32 and not flags[2459001.0] & 64
    assert flags[2459002.0] & 64 and not flags[2459002.0] & 32


def test_next_run_retries_only_failed_downloads(tmp_path, mock_config):
    """A cached band re-downloads just its failed epochs; unavailable ones are left alone."""
    _run_downloads(
        tmp_path, mock_config, _three_epochs(), _downloads(unavailable={2459001.0}, failing={2459002.0})
    )

    result, mock_dl = _run_downloads(
        tmp_path, mock_config, _three_epochs(), _downloads(unavailable={2459001.0})
    )
    assert [float(c.args[0]) for c in mock_dl.call_args_list] == [2459002.0]  # only the failed one
    flags = result["g"].df.set_index("obsjd")["flags"]
    assert flags[2459002.0] == 0  # now measured
    assert flags[2459001.0] & 32  # still unavailable, carried over from the cache
    assert flags[2459000.0] == 0  # reused from the cache

    # With nothing left to retry, the next run is a pure cache hit.
    _, mock_dl = _run_downloads(tmp_path, mock_config, _three_epochs(), _downloads(unavailable={2459001.0}))
    mock_dl.assert_not_called()


def test_retry_unavailable_rechecks_missing_files(tmp_path, mock_config):
    """retry_unavailable=True re-requests epochs IRSA previously did not serve."""
    _run_downloads(tmp_path, mock_config, _three_epochs(), _downloads(unavailable={2459001.0}))

    result, mock_dl = _run_downloads(
        tmp_path, mock_config, _three_epochs(), _downloads(), retry_unavailable=True
    )
    assert [float(c.args[0]) for c in mock_dl.call_args_list] == [2459001.0]
    assert (result["g"].df["flags"] == 0).all()


def test_processing_error_epoch_is_kept(tmp_path, mock_config):
    """An epoch whose image cannot be read stays as a FLAG_PROCESSING_ERROR row dated from metadata."""
    from ztforce.pipeline import run_forced_photometry

    failure = dict(
        flux=np.nan,
        flux_err=np.nan,
        mag=np.nan,
        mag_err=np.nan,
        chisq=np.nan,
        flags=2,
        x_fit=np.nan,
        y_fit=np.nan,
        obsjd=np.nan,
        zero_point=np.nan,
        mag_limit=None,
        image_id="468-3-2-2459000.000",
        band="g",
    )
    with (
        mock.patch("ztforce.pipeline.query_sci_metadata_bands", return_value={"g": _meta(2459000.0)}),
        mock.patch("ztforce.pipeline.download_fits"),
        mock.patch("ztforce.pipeline.download_psf_sidecar"),
        mock.patch("ztforce.pipeline.build_sci_url", return_value="http://fake/url"),
        mock.patch("ztforce.pipeline._process_one_epoch", return_value=failure),
        pytest.warns(UserWarning, match="1 processing error"),
    ):
        result = run_forced_photometry(
            150.0, 2.0, bands=["g"], data_dir=tmp_path / "cache", config=mock_config, show_progress=False
        )
    row = result["g"].df.iloc[0]
    assert row["obsjd"] == 2459000.0
    assert row["flags"] & 2


def test_back_to_back_exposures_are_measured_from_their_own_files(tmp_path, mock_config):
    """Two exposures of one field 40 s apart get separate files, ids and measurements."""
    from ztforce.pipeline import run_forced_photometry

    rows = pd.concat(
        [
            _meta(2459000.50000).assign(filefracday="20200601000000"),
            _meta(2459000.50046).assign(filefracday="20200601000460"),  # +40 s
        ],
        ignore_index=True,
    )

    def _write(url, dest, config):
        dest.write_text(url)  # each file records which exposure it came from
        return dest

    def _fit(fits_path, psf_path, ra, dec, band, image_id, config, full_crpix=None):
        res = _fake_result(band, obsjd=float(image_id.rsplit("-", 1)[1]))
        res["image_id"] = image_id
        res["flux"] = float(Path(fits_path).read_text())  # which file was actually fitted
        assert Path(psf_path).read_text() == Path(fits_path).read_text()  # matching PSF
        return res

    with (
        mock.patch("ztforce.pipeline.query_sci_metadata_bands", return_value={"g": rows}),
        mock.patch("ztforce.pipeline.download_fits", side_effect=_write),
        mock.patch("ztforce.pipeline.download_psf_sidecar", side_effect=_write),
        mock.patch("ztforce.pipeline.build_sci_url", side_effect=lambda row, *a, **k: row["filefracday"]),
        mock.patch("ztforce.pipeline._process_one_epoch", side_effect=_fit),
    ):
        result = run_forced_photometry(
            150.0, 2.0, bands=["g"], data_dir=tmp_path / "cache", config=mock_config, show_progress=False
        )

    df = result["g"].df
    assert (df[["field", "ccdid", "qid"]].to_numpy() == [468, 3, 2]).all()  # group columns
    assert df["image_id"].nunique() == 2
    assert sorted(df["flux"]) == [20200601000000.0, 20200601000460.0]
    assert (df["flags"] == 0).all()


# ── cache key and staleness ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name", ["SNT", "SNU", "BAD_CALIBRATION_INFOBITS", "MAX_SCISIGPIX_DN", "MAX_SEEING_ARCSEC"]
)
def test_cache_key_changes_with_stored_result_settings(monkeypatch, mock_config, name):
    """Thresholds baked into stored detections, limits and flags invalidate the cache."""
    import ztforce.pipeline as pipeline

    before = pipeline._cache_key(mock_config, None)
    monkeypatch.setattr(pipeline, name, getattr(pipeline, name) * 2)
    assert pipeline._cache_key(mock_config, None) != before


def _write_cached(tmp_path, mock_config, queried_at):
    from ztforce.cache import lightcurve_path, make_cache
    from ztforce.lightcurve import Lightcurve
    from ztforce.pipeline import _cache_key

    lc = Lightcurve(ra=150.0, dec=2.0)
    lc.cache_key = _cache_key(mock_config, None)
    lc.queried_at = queried_at
    lc.add_epoch(2459000.0, "g", 1000.0, 50.0, 18.8, 0.05, 26.3, 0)
    lc.save(lightcurve_path(make_cache(tmp_path / "cache"), 150.0, 2.0, "g"))


def _load_cached(tmp_path, mock_config):
    from ztforce.pipeline import run_forced_photometry

    return run_forced_photometry(
        150.0, 2.0, bands=["g"], data_dir=tmp_path / "cache", config=mock_config, show_progress=False
    )


def test_old_cache_warns(tmp_path, mock_config):
    """A cached lightcurve queried over a month ago warns that it misses newer epochs."""
    from datetime import datetime, timedelta, timezone

    old = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat(timespec="seconds")
    _write_cached(tmp_path, mock_config, old)
    with pytest.warns(UserWarning, match="45 days old"):
        result = _load_cached(tmp_path, mock_config)
    assert len(result["g"]) == 1


def test_fresh_cache_does_not_warn(tmp_path, mock_config):
    """A recently queried cache loads silently."""
    from datetime import datetime, timezone

    _write_cached(tmp_path, mock_config, datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _load_cached(tmp_path, mock_config)


def test_new_lightcurve_records_query_time(tmp_path, mock_config):
    """A freshly computed lightcurve stores when its metadata was queried, and it round-trips."""
    from ztforce.cache import lightcurve_path, make_cache
    from ztforce.lightcurve import Lightcurve

    result, _ = _run_downloads(tmp_path, mock_config, _three_epochs(), _downloads())
    assert result["g"].queried_at
    saved = Lightcurve.load(lightcurve_path(make_cache(tmp_path / "cache"), 150.0, 2.0, "g"))
    assert saved.queried_at == result["g"].queried_at


# ── run_forced_photometry_batch ───────────────────────────────────────────────


def test_batch_delegates_to_run_per_target(mock_config):
    """run_forced_photometry_batch calls run_forced_photometry once per target."""
    from ztforce.pipeline import run_forced_photometry_batch

    targets = [
        SkyCoord(ra=150.0, dec=2.0, unit="deg"),
        SkyCoord(ra=200.0, dec=-5.0, unit="deg"),
    ]

    with mock.patch("ztforce.pipeline.run_forced_photometry", return_value={}) as mock_rfp:
        result = run_forced_photometry_batch(targets, bands=["g"], config=mock_config, show_progress=False)

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
        run_forced_photometry_batch(targets, bands=["g"], config=mock_config, show_progress=False)

    assert calls[0][0] == pytest.approx(123.456, rel=1e-5)
    assert calls[0][1] == pytest.approx(-7.89, rel=1e-5)


def test_batch_preserves_target_order(mock_config):
    """run_forced_photometry_batch returns results in the same order as targets."""
    from ztforce.pipeline import run_forced_photometry_batch

    targets = [SkyCoord(ra=10.0 * i, dec=0.0, unit="deg") for i in range(1, 6)]
    expected_ras = [float(c.ra.deg) for c in targets]

    captured_ras = []

    def _capture_order(**kwargs):
        captured_ras.append(kwargs["ra"])
        return {"ra": kwargs["ra"]}

    with mock.patch("ztforce.pipeline.run_forced_photometry", side_effect=_capture_order):
        results = run_forced_photometry_batch(
            targets, bands=["g"], config=mock_config, n_workers=4, show_progress=False
        )

    result_ras = [r["ra"] for r in results]
    assert result_ras == pytest.approx(expected_ras, rel=1e-5)
