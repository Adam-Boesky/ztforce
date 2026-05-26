"""ZTF IRSA metadata queries, URL construction, FITS/PSF download with retry."""

from __future__ import annotations

import random
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import requests
from astropy.io import fits
from ztfquery import buildurl
from ztfquery import query as zquery

from .cache import CacheConfig, fits_path, psf_path
from .config import ZTForceConfig
from .exceptions import FITSDownloadError, NoImagesFoundError

_IRSA_BASE = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci"

_BAND_TO_FILTERCODE = {"g": "zg", "r": "zr", "i": "zi"}


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
    return df.sort_values("obsjd").reset_index(drop=True)


def build_sci_url(row: pd.Series, suffix: str = "sciimg.fits") -> str:
    """Construct the IRSA IBE URL for a science image product."""
    ff = str(int(row["filefracday"]))
    year, month, day, fracday = ff[:4], ff[4:6], ff[6:8], ff[8:]
    paddedfield = str(int(row["field"])).zfill(6)
    filtercode = row["filtercode"]
    paddedccdid = str(int(row["ccdid"])).zfill(2)
    qid = str(int(row["qid"]))
    return buildurl.science_path(
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


def _validate_fits(path: Path) -> bool:
    """Return True if the FITS file opens without error."""
    try:
        with fits.open(str(path), checksum=True) as hdul:
            _ = hdul[0].data
        return True
    except Exception:
        pass
    try:
        result = subprocess.run(["fitscheck", str(path)], capture_output=True, timeout=30)
        return result.returncode == 0
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
                timeout=120,
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
    """Download a FITS file, checking the cache first."""
    if dest.exists() and _validate_fits(dest):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    return _download_with_retry(url, dest, config, validate=True)


def download_psf_sidecar(url: str, dest: Path, config: ZTForceConfig) -> Path:
    """Download a DAOPhot PSF sidecar (.psf) file, checking the cache first."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    return _download_with_retry(url, dest, config, validate=False)


def iter_sci_images(
    ra: float,
    dec: float,
    band: str,
    cache: CacheConfig,
    config: ZTForceConfig,
) -> Iterator[tuple[pd.Series, Path, Path]]:
    """Yield (metadata_row, fits_path, psf_sidecar_path) for every available epoch.

    Files are downloaded only when not already cached. Yields lazily.
    """
    df = query_sci_metadata(ra, dec, band, config)
    for _, row in df.iterrows():
        field = int(row["field"])
        ccdid = int(row["ccdid"])
        qid = int(row["qid"])
        obsjd = float(row["obsjd"])

        local_fits = fits_path(cache, field, ccdid, qid, band, obsjd)
        local_psf = psf_path(cache, field, ccdid, qid, band, obsjd)

        fits_url = build_sci_url(row, suffix="sciimg.fits")
        psf_url = build_sci_url(row, suffix="sciimgdao.psf")

        try:
            download_fits(fits_url, local_fits, config)
            download_psf_sidecar(psf_url, local_psf, config)
        except FITSDownloadError:
            continue

        yield row, local_fits, local_psf
