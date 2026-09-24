"""Tests for ztforce.ztf_images (URL construction, validation, download, iteration)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

# ── Helpers ───────────────────────────────────────────────────────────────────

_RA = 150.0
_DEC = 2.0


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


def _session(resp_or_exc):
    """Mock requests.Session whose get() returns *resp_or_exc*, or raises it if it is an exception."""
    session = mock.MagicMock()
    if isinstance(resp_or_exc, Exception):
        session.get.side_effect = resp_or_exc
    else:
        session.get.return_value = resp_or_exc
    return session


def _write_valid_fits(path: Path) -> None:
    fits.writeto(str(path), np.zeros((4, 4), dtype=np.float32), overwrite=True)


# ── build_sci_url ─────────────────────────────────────────────────────────────


def test_build_sci_url_contains_field_ccdid_qid():
    """build_sci_url URL encodes field, ccdid, qid, and filtercode."""
    from ztforce.ztf_images import build_sci_url

    url = build_sci_url(_make_row(), _RA, _DEC)
    assert "000468" in url
    assert "zg" in url
    assert "c03" in url
    assert "q2" in url


def test_build_sci_url_fits_has_cutout_params():
    """FITS URL includes IRSA cutout center and size query parameters."""
    from ztforce.ztf_images import build_sci_url

    url = build_sci_url(_make_row(), _RA, _DEC, suffix="sciimg.fits", cutout_size_arcmin=15.0)
    assert f"center={_RA},{_DEC}" in url
    assert "size=" in url
    assert "arcsec" in url


def test_build_sci_url_psf_has_no_cutout_params():
    """PSF sidecar URL has no cutout query parameters."""
    from ztforce.ztf_images import build_sci_url

    url = build_sci_url(_make_row(), _RA, _DEC, suffix="sciimgdao.psf")
    assert "center=" not in url
    assert "size=" not in url


def test_build_sci_url_default_suffix_is_sciimg():
    """Default suffix is sciimg.fits."""
    from ztforce.ztf_images import build_sci_url

    url = build_sci_url(_make_row(), _RA, _DEC)
    assert "sciimg.fits" in url


def test_build_sci_url_cutout_size_scales():
    """Larger cutout_size_arcmin produces a larger size value in the URL."""
    from ztforce.ztf_images import build_sci_url

    url_small = build_sci_url(_make_row(), _RA, _DEC, cutout_size_arcmin=5.0)
    url_large = build_sci_url(_make_row(), _RA, _DEC, cutout_size_arcmin=20.0)
    # Both contain "size=" but the numeric value differs
    assert url_small != url_large
    assert "300.0arcsec" in url_small
    assert "1200.0arcsec" in url_large


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


# ── download_fits ─────────────────────────────────────────────────────────────


def test_download_fits_triggers_request(tmp_path, mock_config):
    """download_fits fetches the URL and writes the result to dest."""
    from ztforce.ztf_images import download_fits

    path = tmp_path / "new.fits"
    _write_valid_fits(tmp_path / "_template.fits")
    valid_bytes = (tmp_path / "_template.fits").read_bytes()

    resp = mock.MagicMock()
    resp.content = valid_bytes
    resp.raise_for_status = mock.MagicMock()

    with mock.patch("ztforce.ztf_images._get_session", return_value=_session(resp)):
        result = download_fits("http://fake/url", path, mock_config)

    assert result == path
    assert path.exists()


# ── download_psf_sidecar ──────────────────────────────────────────────────────


def test_download_psf_sidecar_writes_bytes(tmp_path, mock_config):
    """download_psf_sidecar fetches the URL and writes the result to dest."""
    from ztforce.ztf_images import download_psf_sidecar

    path = tmp_path / "new.psf"
    resp = mock.MagicMock()
    resp.content = b"PSF data"
    resp.raise_for_status = mock.MagicMock()

    with mock.patch("ztforce.ztf_images._get_session", return_value=_session(resp)):
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

    with mock.patch("ztforce.ztf_images._get_session", return_value=_session(resp)):
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
        mock.patch("ztforce.ztf_images._get_session", return_value=_session(Exception("connection refused"))),
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
        mock.patch("ztforce.ztf_images._get_session", return_value=_session(resp)),
        mock.patch("ztforce.ztf_images.time.sleep"),
        pytest.raises(FITSDownloadError),
    ):
        _download_with_retry("http://fake/url", path, mock_config, validate=False)


def test_download_with_retry_closes_response_on_bad_status(tmp_path, mock_config):
    """The response is closed even when raise_for_status fails, so connections aren't leaked."""
    from ztforce.exceptions import FITSDownloadError
    from ztforce.ztf_images import _download_with_retry

    mock_config.max_retries = 1
    mock_config.retry_base_delay = 0.0
    mock_config.retry_jitter = 0.0
    resp = mock.MagicMock()
    resp.raise_for_status.side_effect = Exception("500 Server Error")

    with (
        mock.patch("ztforce.ztf_images._get_session", return_value=_session(resp)),
        mock.patch("ztforce.ztf_images.time.sleep"),
        pytest.raises(FITSDownloadError),
    ):
        _download_with_retry("http://fake/url", tmp_path / "x.psf", mock_config, validate=False)

    resp.close.assert_called_once()


def test_download_with_retry_does_not_retry_missing_file(tmp_path, mock_config):
    """A 404 raises ProductUnavailableError after one request, without backoff."""
    from ztforce.exceptions import ProductUnavailableError
    from ztforce.ztf_images import _download_with_retry

    resp = mock.MagicMock()
    resp.status_code = 404
    session = _session(resp)
    with (
        mock.patch("ztforce.ztf_images._get_session", return_value=session),
        mock.patch("ztforce.ztf_images.time.sleep") as sleep,
        pytest.raises(ProductUnavailableError) as exc,
    ):
        _download_with_retry("http://fake/url", tmp_path / "x.fits", mock_config)

    assert exc.value.status == 404
    assert session.get.call_count == 1
    sleep.assert_not_called()


def test_download_with_retry_retries_server_errors(tmp_path, mock_config):
    """A 5xx is transient: retried up to max_retries, then FITSDownloadError (not unavailable)."""
    from ztforce.exceptions import FITSDownloadError, ProductUnavailableError
    from ztforce.ztf_images import _download_with_retry

    mock_config.max_retries = 3
    resp = mock.MagicMock()
    resp.status_code = 503
    resp.raise_for_status.side_effect = Exception("503 Service Unavailable")
    session = _session(resp)
    with (
        mock.patch("ztforce.ztf_images._get_session", return_value=session),
        mock.patch("ztforce.ztf_images.time.sleep"),
        pytest.raises(FITSDownloadError) as exc,
    ):
        _download_with_retry("http://fake/url", tmp_path / "x.fits", mock_config)

    assert not isinstance(exc.value, ProductUnavailableError)
    assert session.get.call_count == 3


# ── _get_session ──────────────────────────────────────────────────────────────


def test_get_session_reused_within_thread_and_carries_auth(mock_config):
    """Repeated calls on one thread return the same session, authenticated with the config."""
    from ztforce.ztf_images import _get_session

    first = _get_session(mock_config)
    assert _get_session(mock_config) is first
    assert first.auth == ("testuser", "testpass")


def test_get_session_distinct_per_thread(mock_config):
    """Each thread gets its own session (requests.Session is not documented thread-safe)."""
    import threading

    from ztforce.ztf_images import _get_session

    sessions = []
    thread = threading.Thread(target=lambda: sessions.append(_get_session(mock_config)))
    thread.start()
    thread.join()
    assert sessions[0] is not _get_session(mock_config)


# ── query_sci_metadata_bands (single band) ────────────────────────────────────────────────────────


def test_query_sci_metadata_raises_when_no_images(mock_config):
    """query_sci_metadata_bands raises NoImagesFoundError when the metadata search returns nothing."""
    from ztforce.exceptions import NoImagesFoundError
    from ztforce.ztf_images import query_sci_metadata_bands

    fetch = mock.MagicMock(return_value=pd.DataFrame())

    with (
        mock.patch("ztforce.ztf_images._fetch_metadata", fetch),
        pytest.raises(NoImagesFoundError),
    ):
        query_sci_metadata_bands(_RA, _DEC, ["g"], mock_config)["g"]


def test_query_sci_metadata_raises_when_none(mock_config):
    """query_sci_metadata_bands raises NoImagesFoundError when metatable is None."""
    from ztforce.exceptions import NoImagesFoundError
    from ztforce.ztf_images import query_sci_metadata_bands

    fetch = mock.MagicMock(return_value=None)

    with (
        mock.patch("ztforce.ztf_images._fetch_metadata", fetch),
        pytest.raises(NoImagesFoundError),
    ):
        query_sci_metadata_bands(_RA, _DEC, ["g"], mock_config)["g"]


def test_query_sci_metadata_returns_sorted_df(mock_config):
    """query_sci_metadata_bands returns a DataFrame sorted by obsjd ascending."""
    from ztforce.ztf_images import query_sci_metadata_bands

    df = pd.DataFrame(
        [
            {"obsjd": 2459002.0, "field": 1, "ccdid": 1, "qid": 1, "filtercode": "zg", "filefracday": 1},
            {"obsjd": 2459001.0, "field": 1, "ccdid": 1, "qid": 1, "filtercode": "zg", "filefracday": 1},
        ]
    )
    fetch = mock.MagicMock(return_value=df)

    with mock.patch("ztforce.ztf_images._fetch_metadata", fetch):
        result = query_sci_metadata_bands(_RA, _DEC, ["g"], mock_config)["g"]

    assert list(result["obsjd"]) == [2459001.0, 2459002.0]


# ── query_sci_metadata_bands ──────────────────────────────────────────────────


def _metatable(*filtercodes_and_jds):
    return pd.DataFrame(
        [
            {"obsjd": jd, "field": 1, "ccdid": 1, "qid": 1, "filtercode": fc, "filefracday": 1}
            for fc, jd in filtercodes_and_jds
        ]
    )


def test_query_sci_metadata_bands_splits_by_band_in_one_query(mock_config):
    """One IRSA query covers all bands; rows are split per band and sorted by obsjd."""
    from ztforce.ztf_images import query_sci_metadata_bands

    fetch = mock.MagicMock(return_value=_metatable(("zr", 3.0), ("zg", 2.0), ("zg", 1.0), ("zi", 4.0)))

    with mock.patch("ztforce.ztf_images._fetch_metadata", fetch):
        result = query_sci_metadata_bands(_RA, _DEC, ["g", "r", "i"], mock_config)

    fetch.assert_called_once()
    # All three filters requested: no filter clause is sent at all.
    assert "WHERE=&" in fetch.call_args.args[0]
    # A point search for exposures that contain the target, not an area overlap.
    assert fetch.call_args.args[0].endswith("&INTERSECT=CENTER")
    assert "SIZE=" not in fetch.call_args.args[0]
    assert list(result) == ["g", "r", "i"]
    assert list(result["g"]["obsjd"]) == [1.0, 2.0]
    assert list(result["r"]["obsjd"]) == [3.0]


def test_query_sci_metadata_bands_subset_uses_in_clause(mock_config):
    """A subset of bands is filtered server-side with an IN clause."""
    from ztforce.ztf_images import query_sci_metadata_bands

    fetch = mock.MagicMock(return_value=_metatable(("zg", 1.0), ("zi", 2.0)))

    with mock.patch("ztforce.ztf_images._fetch_metadata", fetch):
        query_sci_metadata_bands(_RA, _DEC, ["g", "i"], mock_config)

    assert "WHERE=filtercode+IN+('zg','zi')" in fetch.call_args.args[0]


def test_query_sci_metadata_bands_omits_empty_bands(mock_config):
    """Bands with no rows are left out of the result rather than returned empty."""
    from ztforce.ztf_images import query_sci_metadata_bands

    fetch = mock.MagicMock(return_value=_metatable(("zg", 1.0)))

    with mock.patch("ztforce.ztf_images._fetch_metadata", fetch):
        result = query_sci_metadata_bands(_RA, _DEC, ["g", "i"], mock_config)

    assert list(result) == ["g"]


def test_query_sci_metadata_bands_raises_when_no_requested_band(mock_config):
    """NoImagesFoundError when the rows returned contain none of the requested bands."""
    from ztforce.exceptions import NoImagesFoundError
    from ztforce.ztf_images import query_sci_metadata_bands

    fetch = mock.MagicMock(return_value=_metatable(("zr", 1.0)))

    with (
        mock.patch("ztforce.ztf_images._fetch_metadata", fetch),
        pytest.raises(NoImagesFoundError),
    ):
        query_sci_metadata_bands(_RA, _DEC, ["g"], mock_config)


def test_fetch_metadata_uses_timeout(mock_config):
    """The metadata search runs through the shared session with a timeout (ztfquery has none)."""
    from ztforce.ztf_images import _METADATA_TIMEOUT_SEC, _fetch_metadata

    resp = mock.MagicMock()
    resp.text = "obsjd,field\n2459000.5,468\n"
    session = _session(resp)
    with mock.patch("ztforce.ztf_images._get_session", return_value=session):
        df = _fetch_metadata("http://fake/search", mock_config)

    assert session.get.call_args.kwargs["timeout"] == _METADATA_TIMEOUT_SEC
    assert list(df.columns) == ["obsjd", "field"] and len(df) == 1
    resp.close.assert_called_once()
