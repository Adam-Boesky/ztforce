"""ZTF IRSA metadata queries, URL construction, FITS/PSF download with retry."""

from __future__ import annotations

import io
import random
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import requests
from astropy.io import fits
from ztfquery import buildurl, metasearch

from ._constants import DEFAULT_CUTOUT_SIZE_ARCMIN
from .config import ZTForceConfig
from .exceptions import FITSDownloadError, NoImagesFoundError, ProductUnavailableError

_IRSA_BASE = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products"
_DOWNLOAD_TIMEOUT_SEC = 120
_METADATA_TIMEOUT_SEC = 180  # the spatial search itself can take ~30 s
_PERMANENT_HTTP_ERRORS = frozenset({401, 403, 404, 410})

_BAND_TO_FILTERCODE = {"g": "zg", "r": "zr", "i": "zi"}
_REQUIRED_METADATA_COLS = {"obsjd", "field", "ccdid", "qid", "filtercode", "filefracday"}


def query_sci_metadata_bands(
    ra: float,
    dec: float,
    bands: Sequence[str],
    config: ZTForceConfig,
) -> dict[str, pd.DataFrame]:
    """Query ZTF IRSA once for the science exposures whose footprint contains (ra, dec).

    IRSA's cost is the spatial search, not the band filter, so one query for every
    band takes about as long as a query for any single band.

    Returns a dict mapping band to a DataFrame sorted by obsjd ascending.  Bands with
    no exposures are omitted.  Raises NoImagesFoundError when no band has any.
    """
    filtercodes = [_BAND_TO_FILTERCODE[b] for b in bands]
    # Every filter requested: no clause at all, so IRSA does no filtering work.
    if set(filtercodes) == set(_BAND_TO_FILTERCODE.values()):
        sql_query = None
    else:
        sql_query = "filtercode IN (" + ",".join(f"'{fc}'" for fc in filtercodes) + ")"
    desc = f"ZTF {'/'.join(bands)}-band"

    last_exc: Exception | None = None
    for attempt in range(config.max_retries):
        try:
            # A point search for footprints containing the target: an area search also
            # returns exposures where it falls just off the CCD, which cannot be measured.
            url = metasearch.build_query(kind="sci", radec=(ra, dec), sql_query=sql_query, ct="csv")
            df = _fetch_metadata(url + "&INTERSECT=CENTER", config)
            if df is None or df.empty:
                raise NoImagesFoundError(f"No {desc} science images found at ({ra:.5f}, {dec:.5f}).")
            if not _REQUIRED_METADATA_COLS.issubset(df.columns):
                # Service returned garbage (e.g. HTML error page) — treat as transient and retry.
                raise RuntimeError(
                    f"IRSA metadata query returned unexpected response "
                    f"(columns: {list(df.columns)[:5]}). "
                    f"The service may be temporarily unavailable."
                )
            by_band = {}
            for band, fc in zip(bands, filtercodes, strict=True):
                sub = df[df["filtercode"] == fc]
                if not sub.empty:
                    by_band[band] = sub.sort_values("obsjd").reset_index(drop=True)
            if not by_band:
                raise NoImagesFoundError(f"No {desc} science images found at ({ra:.5f}, {dec:.5f}).")
            return by_band
        except NoImagesFoundError:
            raise
        except Exception as exc:
            last_exc = exc
            delay = config.retry_base_delay * (2**attempt) + random.uniform(0, config.retry_jitter)
            time.sleep(delay)
    raise NoImagesFoundError(
        f"IRSA metadata query failed after {config.max_retries} attempts "
        f"for {desc} at ({ra:.5f}, {dec:.5f}): {last_exc}"
    )


def _fetch_metadata(url: str, config: ZTForceConfig) -> pd.DataFrame:
    """Run an IBE metadata search (URL from ztfquery) with a timeout.

    Same query and table as ``ztfquery``'s ``load_metadata``, but through the shared
    session and with a timeout, so a stalled connection fails and is retried instead
    of hanging the worker.
    """
    resp = _get_session(config).get(url, timeout=_METADATA_TIMEOUT_SEC)
    try:
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    finally:
        resp.close()


def build_sci_url(
    row: pd.Series,
    ra: float,
    dec: float,
    suffix: str = "sciimg.fits",
    cutout_size_arcmin: float = DEFAULT_CUTOUT_SIZE_ARCMIN,
) -> str:
    """Construct the IRSA IBE URL for a science image product.

    For FITS files a cutout query is appended so that only a
    ``cutout_size_arcmin`` × ``cutout_size_arcmin`` region centred on
    ``(ra, dec)`` is downloaded.  PSF sidecar files are returned in full
    (they are small text files that describe the whole quadrant).
    """
    ff = str(int(row["filefracday"]))
    year, month, day, fracday = ff[:4], ff[4:6], ff[6:8], ff[8:]
    paddedfield = str(int(row["field"])).zfill(6)
    filtercode = row["filtercode"]
    paddedccdid = str(int(row["ccdid"])).zfill(2)
    qid = str(int(row["qid"]))
    base = buildurl.science_path(
        year=year,
        month=month,
        day=day,
        fracday=fracday,
        paddedfield=paddedfield,
        filtercode=filtercode,
        paddedccdid=paddedccdid,
        qid=qid,
        suffix=suffix,
        source=_IRSA_BASE,
    )
    if suffix.endswith(".fits"):
        size_arcsec = cutout_size_arcmin * 60.0
        return f"{base}?center={ra},{dec}&size={size_arcsec}arcsec"
    return base


def _validate_fits(path: Path) -> bool:
    """Return True if the FITS file opens without error."""
    try:
        with fits.open(str(path), checksum=True) as hdul:
            _ = hdul[0].data
        return True
    except Exception:
        return False


_thread_local = threading.local()


def _get_session(config: ZTForceConfig) -> requests.Session:
    """Return this thread's HTTP session, creating it on first use.

    Reusing a session keeps the TCP/TLS connection to IRSA alive and carries its
    login cookie between requests, which IRSA recommends over authenticating every
    request.  Measured at ~4x faster per file than a fresh ``requests.get``.
    ``requests.Session`` is not documented as thread-safe, so each download thread
    keeps its own.
    """
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    session.auth = (config.irsa_user, config.irsa_pass)
    return session


def _download_with_retry(
    url: str,
    dest: Path,
    config: ZTForceConfig,
    validate: bool = True,
) -> Path:
    """Download *url* to *dest*, retrying on failure with exponential backoff.

    Raises :class:`ProductUnavailableError` at once, without retrying, when IRSA says
    the file is not there or not ours to read (401/403/404/410): IRSA's metadata lists
    some exposures whose files its server does not hold.
    """
    for attempt in range(config.max_retries):
        try:
            resp = _get_session(config).get(url, timeout=_DOWNLOAD_TIMEOUT_SEC)
            try:
                if resp.status_code in _PERMANENT_HTTP_ERRORS:
                    raise ProductUnavailableError(url, resp.status_code)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
            finally:
                # Hand the connection back to the pool even when the request failed.
                resp.close()
            if not validate or _validate_fits(dest):
                return dest
            dest.unlink(missing_ok=True)
        except ProductUnavailableError:
            raise
        except Exception:
            pass
        delay = config.retry_base_delay * (2**attempt) + random.uniform(0, config.retry_jitter)
        time.sleep(delay)
    raise FITSDownloadError(f"Failed to download {url} after {config.max_retries} attempts.")


def download_fits(url: str, dest: Path, config: ZTForceConfig) -> Path:
    """Download a FITS cutout to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    return _download_with_retry(url, dest, config, validate=True)


def download_psf_sidecar(url: str, dest: Path, config: ZTForceConfig) -> Path:
    """Download a DAOPhot PSF sidecar (.psf) file to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    return _download_with_retry(url, dest, config, validate=False)
