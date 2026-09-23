"""Orchestration: forced PSF photometry with source-level batch parallelism."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import tempfile
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
from astropy.coordinates import SkyCoord
from tqdm.auto import tqdm

from ._constants import (
    _PHOTOMETRY_VERSION,
    BAD_CALIBRATION_INFOBITS,
    FLAG_BAD_CALIBRATION,
    FLAG_BAD_SEEING,
    FLAG_NOISY_IMAGE,
    FLAG_PROCESSING_ERROR,
    MAX_SCISIGPIX_DN,
    MAX_SEEING_ARCSEC,
)
from .cache import lightcurve_path, make_cache
from .config import ZTForceConfig, build_config
from .exceptions import NoImagesFoundError
from .image import ZTFImage
from .lightcurve import Lightcurve
from .psf import forced_phot_at_position, parse_daophot_psf
from .ztf_images import build_sci_url, download_fits, download_psf_sidecar, query_sci_metadata_bands

# ── Cache key ────────────────────────────────────────────────────────────────


def _cache_key(config: ZTForceConfig, max_epochs: int | None) -> str:
    """12-hex-char hash of the parameters that affect photometry output."""
    params = {
        "photometry_version": _PHOTOMETRY_VERSION,
        "cutout_size_arcmin": config.cutout_size_arcmin,
        "default_gain": config.default_gain,
        "max_epochs": max_epochs,
    }
    blob = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# ── Per-epoch workers ────────────────────────────────────────────────────────


def _download_epoch(
    row: pd.Series,
    tmp_dir: Path,
    ra: float,
    dec: float,
    config: ZTForceConfig,
) -> tuple[pd.Series, Path, Path]:
    """Download the FITS cutout and PSF sidecar for one epoch.

    Raises on failure so the caller can skip this epoch.
    """
    obsjd = float(row["obsjd"])
    stem = f"{int(row['field'])}-{int(row['ccdid'])}-{int(row['qid'])}-{obsjd:.3f}"
    local_fits = tmp_dir / f"{stem}.fits"
    local_psf = tmp_dir / f"{stem}.psf"
    fits_url = build_sci_url(row, ra, dec, suffix="sciimg.fits", cutout_size_arcmin=config.cutout_size_arcmin)
    psf_url = build_sci_url(row, ra, dec, suffix="sciimgdao.psf")
    download_fits(fits_url, local_fits, config)
    download_psf_sidecar(psf_url, local_psf, config)
    return row, local_fits, local_psf


def _process_one_epoch(
    fits_fpath: str,
    psf_fpath: str,
    ra: float,
    dec: float,
    band: str,
    image_id: str,
    config: ZTForceConfig,
) -> dict:
    """Run forced PSF photometry for one epoch. Returns a result dict."""
    try:
        img = ZTFImage(fits_fpath, band, config)
        parsed_psf = parse_daophot_psf(psf_fpath)
        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        result = forced_phot_at_position(img, parsed_psf, coord)
        result["obsjd"] = img.obs_jd
        result["zero_point"] = img.zero_point
        result["mag_limit"] = img.mag_limit
        result["image_id"] = image_id
        result["band"] = band
        result["infobits"] = img.infobits
        result["seeing"] = img.seeing_arcsec
        result["scisigpix"] = img.scisigpix
        result["flags"] |= _quality_flags(result)
    except Exception:
        result = dict(
            flux=float("nan"),
            flux_err=float("nan"),
            mag=float("nan"),
            mag_err=float("nan"),
            chisq=float("nan"),
            flags=FLAG_PROCESSING_ERROR,
            x_fit=float("nan"),
            y_fit=float("nan"),
            obsjd=float("nan"),
            zero_point=float("nan"),
            mag_limit=None,
            image_id=image_id,
            band=band,
        )
        traceback.print_exc()
    return result


def _quality_flags(result: dict) -> int:
    """Quality-cut bits for one epoch, per the ZFPS user guide section 6.1."""
    flags = 0
    if result["infobits"] >= BAD_CALIBRATION_INFOBITS:
        flags |= FLAG_BAD_CALIBRATION
    if result["scisigpix"] > MAX_SCISIGPIX_DN:
        flags |= FLAG_NOISY_IMAGE
    if result["seeing"] > MAX_SEEING_ARCSEC:
        flags |= FLAG_BAD_SEEING
    return flags


# ── Public API ────────────────────────────────────────────────────────────────


def run_forced_photometry(
    ra: float,
    dec: float,
    bands: tuple[str, ...] | list[str] = ("g", "r", "i"),
    data_dir: str | Path | None = None,
    config: ZTForceConfig | None = None,
    max_epochs: int | None = None,
    force_recompute: bool = False,
    show_progress: bool = True,
    download_workers: int = 8,
    _tqdm_position: int = 0,
    _tqdm_leave: bool = True,
    _download_executor: ThreadPoolExecutor | None = None,
) -> dict[str, Lightcurve]:
    """Run forced PSF photometry at (ra, dec) for all requested bands.

    Downloads ZTF science image cutouts from IRSA, fits the source amplitude at the
    fixed sky position using the per-image DAOPhot PSF sidecar, and returns calibrated
    AB-magnitude lightcurves.  Results are cached on disk; repeated calls for the same
    position return immediately without any network access.

    Args:
        ra: Right ascension in decimal degrees (J2000).
        dec: Declination in decimal degrees (J2000).
        bands: ZTF bands to process.  Any subset of ``("g", "r", "i")``.
        data_dir: Root directory for the on-disk cache.  Defaults to
            ``~/.ztforce/cache`` when ``None``.
        config: Credentials and runtime settings.  Built from environment
            variables / ``~/.ztforce/config.toml`` when ``None``.
        max_epochs: If set, process only the *most recent* ``max_epochs``
            exposures per band.  Useful for quick tests.
        force_recompute: If ``True``, ignore any cached lightcurve and
            redownload + recompute from scratch, overwriting the cache.
        show_progress: If ``True`` (default), display a tqdm progress bar.
        download_workers: Number of concurrent epoch downloads.  Ignored when
            a shared ``_download_executor`` is supplied by the batch wrapper.

    Returns:
        Dict mapping band label (``"g"``, ``"r"``, ``"i"``) to a
        :class:`~ztforce.Lightcurve`.  Bands with no available images are
        omitted.
    """
    cache = make_cache(data_dir)
    if config is None:
        config = build_config()

    ck = _cache_key(config, max_epochs)
    lightcurves: dict[str, Lightcurve] = {}

    # Serve what the cache can; everything else is fetched together below.
    todo: list[str] = []
    for band in bands:
        lc_fpath = lightcurve_path(cache, ra, dec, band)
        if lc_fpath.exists() and not force_recompute:
            lc = Lightcurve.load(lc_fpath)
            if lc.cache_key == ck:
                if show_progress:
                    tqdm.write(f"({ra:.3f}, {dec:.3f}) [{band}] loaded from cache")
                lightcurves[band] = lc
                continue
            # stale cache (settings changed) — fall through and recompute
        todo.append(band)

    if not todo:
        return lightcurves

    # One metadata query for every band still needed (IRSA's cost is the spatial
    # search, so this is ~1/3 the time of querying band by band).
    try:
        metadata = query_sci_metadata_bands(ra, dec, todo, config)
    except NoImagesFoundError:
        return lightcurves

    if max_epochs is not None:
        metadata = {band: df.tail(max_epochs).reset_index(drop=True) for band, df in metadata.items()}

    # Use a shared executor supplied by the batch wrapper, or own one locally.
    _own_executor = _download_executor is None
    dl_exec = _download_executor or ThreadPoolExecutor(max_workers=download_workers)

    desc_base = f"({ra:.3f}, {dec:.3f})"
    bar = tqdm(
        total=2 * sum(len(df) for df in metadata.values()),
        desc=f"{desc_base} downloading",
        position=_tqdm_position,
        leave=_tqdm_leave,
        disable=not show_progress,
        unit="step",
    )
    futures_by_band: dict[str, list[Future]] = {}

    try:
        with tempfile.TemporaryDirectory() as _tmp:
            tmp_dir = Path(_tmp)

            # Download phase: every epoch of every band is submitted up front, so the
            # pool never idles between bands.  The pool is FIFO, so bands finish in
            # order and each is fitted and cached while the next is still downloading.
            for band, df in metadata.items():
                futures_by_band[band] = [
                    dl_exec.submit(_download_epoch, row, tmp_dir, ra, dec, config) for _, row in df.iterrows()
                ]

            for band, futures in futures_by_band.items():
                image_triples: list[tuple[pd.Series, Path, Path]] = []
                for fut in as_completed(futures):
                    with contextlib.suppress(Exception):
                        image_triples.append(fut.result())
                    bar.update(1)

                if not image_triples:
                    continue

                # PSF photometry phase: sequential (CPU-fast, ~15 ms/epoch)
                bar.set_description(f"{desc_base} [{band}] fitting PSF")
                results = []
                for row, fits_p, psf_p in image_triples:
                    image_id = (
                        f"{int(row['field'])}-{int(row['ccdid'])}-{int(row['qid'])}-{float(row['obsjd']):.3f}"
                    )
                    results.append(
                        _process_one_epoch(str(fits_p), str(psf_p), ra, dec, band, image_id, config)
                    )
                    # Done with this epoch; don't hold every band's images on disk at once.
                    fits_p.unlink(missing_ok=True)
                    psf_p.unlink(missing_ok=True)
                    bar.update(1)
                bar.set_description(f"{desc_base} downloading")

                results.sort(key=lambda d: d.get("obsjd", 0))

                # Assemble lightcurve
                lc = Lightcurve(ra=ra, dec=dec)
                for res in results:
                    if not res.get("obsjd") or math.isnan(res.get("obsjd", float("nan"))):
                        continue
                    lc.add_epoch(
                        obsjd=res["obsjd"],
                        band=band,
                        flux=res["flux"],
                        flux_err=res["flux_err"],
                        mag=res["mag"],
                        mag_err=res["mag_err"],
                        zero_point=res["zero_point"],
                        flags=res["flags"],
                        x_fit=res.get("x_fit"),
                        y_fit=res.get("y_fit"),
                        mag_limit=res.get("mag_limit"),
                        image_id=res.get("image_id"),
                        chisq=res.get("chisq"),
                        infobits=res.get("infobits"),
                        seeing=res.get("seeing"),
                        scisigpix=res.get("scisigpix"),
                    )

                lc.cache_key = ck
                lc.save(lightcurve_path(cache, ra, dec, band))
                lightcurves[band] = lc

    finally:
        bar.close()
        # On an early exit, don't leave this target's downloads queued in a shared pool.
        for futures in futures_by_band.values():
            for fut in futures:
                fut.cancel()
        if _own_executor:
            dl_exec.shutdown(wait=False)

    return lightcurves


def run_forced_photometry_batch(
    targets: list[SkyCoord],
    bands: tuple[str, ...] | list[str] = ("g", "r", "i"),
    data_dir: str | Path | None = None,
    config: ZTForceConfig | None = None,
    n_workers: int = 4,
    download_workers: int = 8,
    show_progress: bool = True,
) -> list[dict[str, Lightcurve]]:
    """Run forced photometry for a list of SkyCoord targets in parallel.

    Each target is processed by a dedicated thread; results are returned in the
    same order as ``targets``.  A single shared download thread pool (capped at
    ``download_workers``) is used across all active source workers so that
    concurrency is bounded at one level only.

    Args:
        targets: Sky positions to process.
        bands: ZTF bands to process.  Any subset of ``("g", "r", "i")``.
        data_dir: Root directory for the on-disk cache.
        config: Credentials and runtime settings.
        n_workers: Number of targets to process concurrently.
        download_workers: Total number of concurrent epoch downloads shared
            across all active source workers.
        show_progress: If ``True`` (default), display tqdm progress bars.

    Returns:
        List of band → :class:`~ztforce.Lightcurve` dicts, one per target.
    """
    if config is None:
        config = build_config()

    # Thread-safe pool of tqdm positions 1..n_workers.
    # Position 0 is reserved for the top-level Sources bar.
    _pool_lock = Lock()
    _positions: list[int] = list(range(1, n_workers + 1))

    def _acquire_position() -> int:
        with _pool_lock:
            return _positions.pop(0) if _positions else 0

    def _release_position(pos: int) -> None:
        with _pool_lock:
            if pos > 0:
                _positions.append(pos)
                _positions.sort()

    main_bar = tqdm(
        total=len(targets),
        desc="Sources",
        position=0,
        leave=True,
        disable=not show_progress,
        unit="source",
    )

    # One shared download pool for all source workers combined.
    with ThreadPoolExecutor(max_workers=download_workers) as dl_exec:

        def _run_one(coord: SkyCoord) -> dict[str, Lightcurve]:
            pos = _acquire_position()
            try:
                return run_forced_photometry(
                    ra=float(coord.ra.deg),
                    dec=float(coord.dec.deg),
                    bands=bands,
                    data_dir=data_dir,
                    config=config,
                    show_progress=show_progress,
                    _tqdm_position=pos,
                    _tqdm_leave=False,
                    _download_executor=dl_exec,
                )
            finally:
                _release_position(pos)
                main_bar.update(1)

        results: list[dict[str, Lightcurve]] = [{} for _ in range(len(targets))]
        with ThreadPoolExecutor(max_workers=n_workers) as src_exec:
            future_to_idx = {src_exec.submit(_run_one, coord): i for i, coord in enumerate(targets)}
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()

    main_bar.close()
    return results
