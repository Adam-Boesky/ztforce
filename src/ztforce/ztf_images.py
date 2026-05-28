"""ZTF IRSA metadata queries, URL construction, FITS/PSF download with retry."""

from __future__ import annotations

import random
import time
from pathlib import Path

import pandas as pd
import requests
from astropy.io import fits
from ztfquery import buildurl
from ztfquery import query as zquery

from ._constants import DEFAULT_CUTOUT_SIZE_ARCMIN
from .config import ZTForceConfig
from .exceptions import FITSDownloadError, NoImagesFoundError

_IRSA_BASE = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products"
_DOWNLOAD_TIMEOUT_SEC = 120

_BAND_TO_FILTERCODE = {"g": "zg", "r": "zr", "i": "zi"}
_REQUIRED_METADATA_COLS = {"obsjd", "field", "ccdid", "qid", "filtercode", "filefracday"}


def query_sci_metadata(
    ra: float,
    dec: float,
    band: str,
    config: ZTForceConfig,
    search_radius_deg: float = 0.01,
) -> pd.DataFrame:
    """Query ZTF IRSA for all science exposures covering (ra, dec) in *band*.

    Returns a DataFrame sorted by obsjd ascending.
    Raises NoImagesFoundError when no images are found.
    """
    filtercode = _BAND_TO_FILTERCODE[band]

    last_exc: Exception | None = None
    for attempt in range(config.max_retries):
        try:
            zq = zquery.ZTFQuery()
            zq.load_metadata(
                kind="sci",
                radec=(ra, dec),
                size=search_radius_deg,
                sql_query=f"filtercode='{filtercode}'",
                auth=(config.irsa_user, config.irsa_pass),
            )
            df = zq.metatable
            if df is None or df.empty:
                raise NoImagesFoundError(f"No ZTF {band}-band science images found at ({ra:.5f}, {dec:.5f}).")
            if not _REQUIRED_METADATA_COLS.issubset(df.columns):
                # Service returned garbage (e.g. HTML error page) — treat as transient and retry.
                raise RuntimeError(
                    f"IRSA metadata query returned unexpected response "
                    f"(columns: {list(df.columns)[:5]}). "
                    f"The service may be temporarily unavailable."
                )
            return df.sort_values("obsjd").reset_index(drop=True)
        except NoImagesFoundError:
            raise
        except Exception as exc:
            last_exc = exc
            delay = config.retry_base_delay * (2**attempt) + random.uniform(0, config.retry_jitter)
            time.sleep(delay)
    raise NoImagesFoundError(
        f"IRSA metadata query failed after {config.max_retries} attempts "
        f"for ZTF {band}-band at ({ra:.5f}, {dec:.5f}): {last_exc}"
    )


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


def _download_with_retry(
    url: str,
    dest: Path,
    config: ZTForceConfig,
    validate: bool = True,
) -> Path:
    """Download *url* to *dest*, retrying on failure with exponential backoff."""
    for attempt in range(config.max_retries):
        try:
            resp = requests.get(
                url,
                auth=(config.irsa_user, config.irsa_pass),
                stream=True,
                timeout=_DOWNLOAD_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            if not validate or _validate_fits(dest):
                return dest
            dest.unlink(missing_ok=True)
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
