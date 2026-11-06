"""ZTF IRSA metadata queries, URL construction, FITS/PSF download with retry."""

from __future__ import annotations

import io
import random
import threading
import time
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import pandas as pd
import requests
from astropy.io import fits
from astropy.table import Table
from ztfquery import buildurl, metasearch

from ._constants import DEFAULT_CUTOUT_SIZE_ARCMIN
from .config import ZTForceConfig
from .exceptions import FITSDownloadError, MetadataQueryError, NoImagesFoundError, ProductUnavailableError

_IRSA_BASE = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products"
_DOWNLOAD_TIMEOUT_SEC = 120
_METADATA_TIMEOUT_SEC = 180  # the spatial search itself can take ~30 s
_PERMANENT_HTTP_ERRORS = frozenset({401, 403, 404, 410})

_BAND_TO_FILTERCODE = {"g": "zg", "r": "zr", "i": "zi"}
_REQUIRED_METADATA_COLS = {"obsjd", "field", "ccdid", "qid", "filtercode", "filefracday"}
# The metadata columns the pipeline reads.  Asking for only these (not the full table of
# ~45) halves the search time and cuts the response from ~1.4 MB to ~0.3 MB per position.
METADATA_COLUMNS = (
    "obsjd",
    "field",
    "ccdid",
    "qid",
    "filtercode",
    "filefracday",
    "infobits",
    "seeing",
    "maglimit",
    "crpix1",
    "crpix2",
)
_METADATA_SEARCH_URL = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci"
# Positions per multi-position search: keeps each response (~0.3 MB per position) and
# its server time bounded.
_POSITIONS_PER_QUERY = 50
_POSITIONS_TIMEOUT_SEC = 900


def _band_filter(bands: Sequence[str]) -> tuple[list[str], str | None]:
    """The filtercodes for ``bands`` and the SQL clause selecting them (None for all)."""
    filtercodes = [_BAND_TO_FILTERCODE[b] for b in bands]
    # Every filter requested: no clause at all, so IRSA does no filtering work.
    if set(filtercodes) == set(_BAND_TO_FILTERCODE.values()):
        return filtercodes, None
    return filtercodes, "filtercode IN (" + ",".join(f"'{fc}'" for fc in filtercodes) + ")"


def _split_by_band(
    df: pd.DataFrame, bands: Sequence[str], filtercodes: Sequence[str]
) -> dict[str, pd.DataFrame]:
    """Split metadata rows per band, each sorted by obsjd; bands without rows are omitted."""
    by_band = {}
    for band, fc in zip(bands, filtercodes, strict=True):
        sub = df[df["filtercode"] == fc]
        if not sub.empty:
            by_band[band] = sub.sort_values("obsjd").reset_index(drop=True)
    return by_band


def _search_with_retry(
    fetch: Callable[[], pd.DataFrame | None], required_cols: set[str], desc: str, config: ZTForceConfig
) -> pd.DataFrame:
    """Run a metadata search, retrying with backoff on failures and malformed responses.

    ``fetch`` performs the request and returns a DataFrame.  Raises NoImagesFoundError
    when the search returns no rows, and MetadataQueryError when every attempt failed.
    """
    last_exc: Exception | None = None
    for attempt in range(config.max_retries):
        try:
            df = fetch()
            if df is None or df.empty:
                raise NoImagesFoundError(f"No {desc} science images found.")
            if not required_cols.issubset(df.columns):
                # Service returned garbage (e.g. HTML error page) — treat as transient and retry.
                raise RuntimeError(
                    f"IRSA metadata query returned unexpected response "
                    f"(columns: {list(df.columns)[:5]}). "
                    f"The service may be temporarily unavailable."
                )
            return df
        except NoImagesFoundError:
            raise
        except Exception as exc:
            last_exc = exc
            delay = config.retry_base_delay * (2**attempt) + random.uniform(0, config.retry_jitter)
            time.sleep(delay)
    raise MetadataQueryError(
        f"IRSA metadata query failed after {config.max_retries} attempts for {desc}: {last_exc}"
    )


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
    filtercodes, sql_query = _band_filter(bands)
    desc = f"ZTF {'/'.join(bands)}-band at ({ra:.5f}, {dec:.5f})"
    # A point search for footprints containing the target: an area search also
    # returns exposures where it falls just off the CCD, which cannot be measured.
    url = metasearch.build_query(kind="sci", radec=(ra, dec), sql_query=sql_query, ct="csv")
    url += "&COLUMNS=" + ",".join(METADATA_COLUMNS) + "&INTERSECT=CENTER"
    try:
        df = _search_with_retry(lambda: _fetch_metadata(url, config), _REQUIRED_METADATA_COLS, desc, config)
    except MetadataQueryError as exc:
        # Kept as NoImagesFoundError: a single-position run treats a failed search like
        # an empty one and serves what it has.
        raise NoImagesFoundError(str(exc)) from exc
    by_band = _split_by_band(df, bands, filtercodes)
    if not by_band:
        raise NoImagesFoundError(f"No {desc} science images found.")
    return by_band


def query_sci_metadata_positions(
    positions: Sequence[tuple[float, float]],
    bands: Sequence[str],
    config: ZTForceConfig,
) -> list[dict[str, pd.DataFrame]]:
    """Query ZTF IRSA for the science exposures containing each of several positions.

    IBE takes a table of positions in one request, which costs far less than a spatial
    search per position (20 positions: ~40 s together against ~25 s each).  Positions
    are sent ``_POSITIONS_PER_QUERY`` at a time.

    Returns one band -> DataFrame dict per position, in the order given, each as
    :func:`query_sci_metadata_bands` would return it; a position with no exposures
    gets an empty dict.  Raises NoImagesFoundError when no position has any, and
    MetadataQueryError when a request failed on every attempt (unlike the
    single-position search, a batch must not mistake an outage for empty sky).
    """
    if not positions:
        return []
    filtercodes, sql_query = _band_filter(bands)
    form = {"INTERSECT": "CENTER", "ct": "csv", "COLUMNS": ",".join(METADATA_COLUMNS)}
    if sql_query:
        form["WHERE"] = sql_query

    results: list[dict[str, pd.DataFrame]] = []
    for start in range(0, len(positions), _POSITIONS_PER_QUERY):
        chunk = positions[start : start + _POSITIONS_PER_QUERY]
        desc = f"ZTF {'/'.join(bands)}-band at {len(chunk)} positions"
        files = {"POS": ("positions.tbl", _ipac_position_table(chunk))}
        try:
            df = _search_with_retry(
                partial(_fetch_metadata, _METADATA_SEARCH_URL, config, data=form, files=files),
                _REQUIRED_METADATA_COLS | {"in_row_id"},
                desc,
                config,
            )
        except NoImagesFoundError:
            results.extend({} for _ in chunk)
            continue
        # Each row carries in_row_id: the 1-based row of the uploaded table it matched.
        # (IBE adds in_row_id/in_ra/in_dec even with COLUMNS= set, and honours WHERE as
        # a form field; checked against the service on 2026-11-07.)
        for i in range(len(chunk)):
            sub = df[df["in_row_id"] == i + 1].drop(columns=["in_row_id", "in_ra", "in_dec"], errors="ignore")
            results.append(_split_by_band(sub, bands, filtercodes))
    if not any(results):
        raise NoImagesFoundError(f"No ZTF {'/'.join(bands)}-band science images found at any position.")
    return results


def _ipac_position_table(positions: Sequence[tuple[float, float]]) -> str:
    """An IPAC-format table of ra/dec rows, as IBE's multi-position search expects."""
    table = Table(rows=list(positions), names=("ra", "dec"), dtype=(float, float))
    buf = io.StringIO()
    table.write(buf, format="ascii.ipac")
    return buf.getvalue()


def _fetch_metadata(
    url: str,
    config: ZTForceConfig,
    data: dict | None = None,
    files: dict | None = None,
) -> pd.DataFrame:
    """Run an IBE metadata search with a timeout.

    A GET of ``url`` (built by ztfquery, the same query and table as its
    ``load_metadata``), or a POST when ``data``/``files`` are given (the
    multi-position search).  Through the shared session and with a timeout, so a
    stalled connection fails and is retried instead of hanging the worker.
    """
    session = _get_session(config)
    if data is None and files is None:
        resp = session.get(url, timeout=_METADATA_TIMEOUT_SEC)
    else:
        resp = session.post(url, data=data, files=files, timeout=_POSITIONS_TIMEOUT_SEC)
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
