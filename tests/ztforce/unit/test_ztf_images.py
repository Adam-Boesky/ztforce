"""Tests for ztforce.ztf_images (URL construction, validation, download, iteration)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_row(field=468, ccdid=3, qid=2, obsjd=2459000.0, filtercode="zg"):
    return pd.Series(
        {
            "field": field,
            "ccdid": ccdid,
            "qid": qid,
            "obsjd": obsjd,
            "filtercode": filtercode,
            "filefracday": 20210601123456,
        }
    )


def _write_valid_fits(path: Path) -> None:
    fits.writeto(str(path), np.zeros((4, 4), dtype=np.float32), overwrite=True)


# ── build_sci_url ─────────────────────────────────────────────────────────────


def test_build_sci_url_contains_field_ccdid_qid():
    """build_sci_url URL encodes field, ccdid, qid, and filtercode."""
    from ztforce.ztf_images import build_sci_url

    row = _make_row()
    url = build_sci_url(row)
    assert "000468" in url
    assert "zg" in url
    assert "c03" in url
    assert "q2" in url


def test_build_sci_url_default_suffix_is_sciimg():
    """Default suffix is sciimg.fits."""
    from ztforce.ztf_images import build_sci_url

    url = build_sci_url(_make_row())
    assert url.endswith("sciimg.fits")


def test_build_sci_url_psf_suffix():
    """Passing sciimgdao.psf suffix produces a .psf URL."""
    from ztforce.ztf_images import build_sci_url

    url = build_sci_url(_make_row(), suffix="sciimgdao.psf")
    assert url.endswith("sciimgdao.psf")


def test_build_sci_url_different_suffixes_differ():
    """FITS and PSF URLs differ only in their suffix."""
    from ztforce.ztf_images import build_sci_url

    row = _make_row()
    fits_url = build_sci_url(row, suffix="sciimg.fits")
    psf_url = build_sci_url(row, suffix="sciimgdao.psf")
    assert fits_url != psf_url
    assert fits_url[: -len("sciimg.fits")] == psf_url[: -len("sciimgdao.psf")]


# ── _validate_fits ────────────────────────────────────────────────────────────


def test_validate_fits_valid_file(tmp_path):
    """_validate_fits returns True for a well-formed FITS file."""
    from ztforce.ztf_images import _validate_fits

    path = tmp_path / "good.fits"
    _write_valid_fits(path)
    assert _validate_fits(path) is True


def test_validate_fits_corrupt_file(tmp_path):
    """_validate_fits returns False for a non-FITS file."""
    from ztforce.ztf_images import _validate_fits

    path = tmp_path / "bad.fits"
    path.write_bytes(b"this is not a fits file at all")
    assert _validate_fits(path) is False


def test_validate_fits_empty_file(tmp_path):
    """_validate_fits returns False for a zero-byte file."""
    from ztforce.ztf_images import _validate_fits

    path = tmp_path / "empty.fits"
    path.write_bytes(b"")
    assert _validate_fits(path) is False


# ── download_fits (cache hit) ─────────────────────────────────────────────────


def test_download_fits_cache_hit_skips_request(tmp_path, mock_config):
    """download_fits returns immediately when a valid file is already cached."""
    from ztforce.ztf_images import download_fits

    path = tmp_path / "cached.fits"
    _write_valid_fits(path)

    with mock.patch("ztforce.ztf_images.requests.get") as mock_get:
        result = download_fits("http://fake/url", path, mock_config)

    mock_get.assert_not_called()
    assert result == path


def test_download_fits_cache_miss_triggers_request(tmp_path, mock_config):
    """download_fits fetches the URL when the file is absent."""
    from ztforce.ztf_images import download_fits

    path = tmp_path / "new.fits"
    _write_valid_fits(tmp_path / "_template.fits")
    valid_bytes = (tmp_path / "_template.fits").read_bytes()

    resp = mock.MagicMock()
    resp.content = valid_bytes
    resp.raise_for_status = mock.MagicMock()

    with mock.patch("ztforce.ztf_images.requests.get", return_value=resp):
        result = download_fits("http://fake/url", path, mock_config)

    assert result == path
    assert path.exists()


# ── download_psf_sidecar ──────────────────────────────────────────────────────


def test_download_psf_sidecar_cache_hit(tmp_path, mock_config):
    """download_psf_sidecar skips the network when a non-empty file exists."""
    from ztforce.ztf_images import download_psf_sidecar

    path = tmp_path / "cached.psf"
    path.write_text("PSF content")

    with mock.patch("ztforce.ztf_images.requests.get") as mock_get:
        result = download_psf_sidecar("http://fake/url", path, mock_config)

    mock_get.assert_not_called()
    assert result == path


def test_download_psf_sidecar_empty_redownloads(tmp_path, mock_config):
    """download_psf_sidecar re-downloads when cached file is zero bytes."""
    from ztforce.ztf_images import download_psf_sidecar

    path = tmp_path / "stale.psf"
    path.write_bytes(b"")  # zero-byte → stale

    resp = mock.MagicMock()
    resp.content = b"PSF data"
    resp.raise_for_status = mock.MagicMock()

    with mock.patch("ztforce.ztf_images.requests.get", return_value=resp):
        download_psf_sidecar("http://fake/url", path, mock_config)

    assert path.read_bytes() == b"PSF data"


# ── _download_with_retry ──────────────────────────────────────────────────────


def test_download_with_retry_success(tmp_path, mock_config):
    """_download_with_retry writes the response bytes and returns the path."""
    from ztforce.ztf_images import _download_with_retry

    path = tmp_path / "out.psf"
    resp = mock.MagicMock()
    resp.content = b"some psf bytes"
    resp.raise_for_status = mock.MagicMock()

    with mock.patch("ztforce.ztf_images.requests.get", return_value=resp):
        result = _download_with_retry("http://fake/url", path, mock_config, validate=False)

    assert result == path
    assert path.read_bytes() == b"some psf bytes"


def test_download_with_retry_exhausted_raises(tmp_path, mock_config):
    """_download_with_retry raises FITSDownloadError after all retries fail."""
    from ztforce.exceptions import FITSDownloadError
    from ztforce.ztf_images import _download_with_retry

    path = tmp_path / "fail.fits"
    mock_config.max_retries = 2
    mock_config.retry_base_delay = 0.0
    mock_config.retry_jitter = 0.0

    with (
        mock.patch("ztforce.ztf_images.requests.get", side_effect=Exception("connection refused")),
        mock.patch("ztforce.ztf_images.time.sleep"),
        pytest.raises(FITSDownloadError),
    ):
        _download_with_retry("http://fake/url", path, mock_config, validate=False)


def test_download_with_retry_retries_on_bad_status(tmp_path, mock_config):
    """_download_with_retry retries when raise_for_status raises."""
    from ztforce.exceptions import FITSDownloadError
    from ztforce.ztf_images import _download_with_retry

    path = tmp_path / "fail.fits"
    mock_config.max_retries = 2
    mock_config.retry_base_delay = 0.0
    mock_config.retry_jitter = 0.0

    resp = mock.MagicMock()
    resp.raise_for_status.side_effect = Exception("403 Forbidden")

    with (
        mock.patch("ztforce.ztf_images.requests.get", return_value=resp),
        mock.patch("ztforce.ztf_images.time.sleep"),
        pytest.raises(FITSDownloadError),
    ):
        _download_with_retry("http://fake/url", path, mock_config, validate=False)


# ── query_sci_metadata ────────────────────────────────────────────────────────


def test_query_sci_metadata_raises_when_no_images(mock_config):
    """query_sci_metadata raises NoImagesFoundError when ZTFQuery returns empty."""
    from ztforce.exceptions import NoImagesFoundError
    from ztforce.ztf_images import query_sci_metadata

    mock_zq = mock.MagicMock()
    mock_zq.metatable = pd.DataFrame()

    with (
        mock.patch("ztforce.ztf_images.zquery.ZTFQuery", return_value=mock_zq),
        pytest.raises(NoImagesFoundError),
    ):
        query_sci_metadata(150.0, 2.0, "g", mock_config)


def test_query_sci_metadata_raises_when_none(mock_config):
    """query_sci_metadata raises NoImagesFoundError when metatable is None."""
    from ztforce.exceptions import NoImagesFoundError
    from ztforce.ztf_images import query_sci_metadata

    mock_zq = mock.MagicMock()
    mock_zq.metatable = None

    with (
        mock.patch("ztforce.ztf_images.zquery.ZTFQuery", return_value=mock_zq),
        pytest.raises(NoImagesFoundError),
    ):
        query_sci_metadata(150.0, 2.0, "g", mock_config)


def test_query_sci_metadata_returns_sorted_df(mock_config):
    """query_sci_metadata returns a DataFrame sorted by obsjd ascending."""
    from ztforce.ztf_images import query_sci_metadata

    df = pd.DataFrame(
        [
            {"obsjd": 2459002.0, "field": 1, "ccdid": 1, "qid": 1},
            {"obsjd": 2459001.0, "field": 1, "ccdid": 1, "qid": 1},
        ]
    )
    mock_zq = mock.MagicMock()
    mock_zq.metatable = df

    with mock.patch("ztforce.ztf_images.zquery.ZTFQuery", return_value=mock_zq):
        result = query_sci_metadata(150.0, 2.0, "g", mock_config)

    assert list(result["obsjd"]) == [2459001.0, 2459002.0]


# ── iter_sci_images ───────────────────────────────────────────────────────────


def test_iter_sci_images_yields_paths(tmp_path, mock_config, cache):
    """iter_sci_images yields (row, fits_path, psf_path) for each image."""
    # Write the files that would be "downloaded" into the cache
    from ztforce.cache import fits_path as fp
    from ztforce.cache import psf_path as pp
    from ztforce.ztf_images import iter_sci_images

    row = _make_row(obsjd=2459000.0)
    lf = fp(cache, int(row["field"]), int(row["ccdid"]), int(row["qid"]), "g", float(row["obsjd"]))
    lp = pp(cache, int(row["field"]), int(row["ccdid"]), int(row["qid"]), "g", float(row["obsjd"]))
    lf.parent.mkdir(parents=True, exist_ok=True)
    lp.parent.mkdir(parents=True, exist_ok=True)
    _write_valid_fits(lf)
    lp.write_text("PSF content")

    df = pd.DataFrame([row])
    with (
        mock.patch("ztforce.ztf_images.query_sci_metadata", return_value=df),
        mock.patch("ztforce.ztf_images.build_sci_url", return_value="http://fake"),
        mock.patch("ztforce.ztf_images.download_fits", return_value=lf),
        mock.patch("ztforce.ztf_images.download_psf_sidecar", return_value=lp),
    ):
        results = list(iter_sci_images(150.0, 2.0, "g", cache, mock_config))

    assert len(results) == 1
    _, out_fits, out_psf = results[0]
    assert out_fits == lf
    assert out_psf == lp


def test_iter_sci_images_skips_failed_downloads(tmp_path, mock_config, cache):
    """iter_sci_images silently skips images whose download raises FITSDownloadError."""
    from ztforce.exceptions import FITSDownloadError
    from ztforce.ztf_images import iter_sci_images

    row = _make_row(obsjd=2459000.0)
    df = pd.DataFrame([row])

    with (
        mock.patch("ztforce.ztf_images.query_sci_metadata", return_value=df),
        mock.patch("ztforce.ztf_images.build_sci_url", return_value="http://fake"),
        mock.patch("ztforce.ztf_images.download_fits", side_effect=FITSDownloadError("fail")),
    ):
        results = list(iter_sci_images(150.0, 2.0, "g", cache, mock_config))

    assert results == []
