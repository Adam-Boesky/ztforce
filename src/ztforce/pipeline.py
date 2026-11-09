"""Orchestration: forced PSF photometry with source-level batch parallelism."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import warnings
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from tqdm.auto import tqdm

from ._constants import (
    _PHOTOMETRY_VERSION,
    BAD_CALIBRATION_INFOBITS,
    FLAG_BAD_CALIBRATION,
    FLAG_BAD_SEEING,
    FLAG_DOWNLOAD_FAILED,
    FLAG_NOISY_IMAGE,
    FLAG_PROCESSING_ERROR,
    FLAG_UNAVAILABLE,
    MAX_SCISIGPIX_DN,
    MAX_SEEING_ARCSEC,
    SKY_ANNULUS_GAP_PX,
    SKY_ANNULUS_WIDTH_PX,
)
from .cache import CacheConfig, lightcurve_path, make_cache
from .config import ZTForceConfig, build_config
from .exceptions import MetadataQueryError, NoImagesFoundError, ProductUnavailableError
from .image import ZTFImage
from .lightcurve import SNT, SNU, Lightcurve
from .psf import forced_phot_at_position, parse_daophot_psf
from .ztf_images import (
    build_sci_url,
    download_fits,
    download_psf_sidecar,
    query_sci_metadata_bands,
    query_sci_metadata_positions,
)

# ── Cache key ────────────────────────────────────────────────────────────────


def _cache_key(config: ZTForceConfig, max_epochs: int | None, measure_flagged: bool = False) -> str:
    """12-hex-char hash of the parameters that affect photometry output."""
    params = {
        "photometry_version": _PHOTOMETRY_VERSION,
        "cutout_size_arcmin": config.cutout_size_arcmin,
        "default_gain": config.default_gain,
        "psf_fit_taper": config.psf_fit_taper,
        "max_epochs": max_epochs,
        "measure_flagged": measure_flagged,
        # Stored per-epoch results (detection, upper_limit, flags) depend on these.
        "snt": SNT,
        "snu": SNU,
        "bad_calibration_infobits": BAD_CALIBRATION_INFOBITS,
        "max_scisigpix_dn": MAX_SCISIGPIX_DN,
        "max_seeing_arcsec": MAX_SEEING_ARCSEC,
        "sky_annulus_px": [SKY_ANNULUS_GAP_PX, SKY_ANNULUS_WIDTH_PX],
    }
    blob = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# A cached lightcurve never picks up epochs taken after its metadata query; warn
# once it is this old.
CACHE_STALE_DAYS = 30


def _warn_if_stale(lc: Lightcurve, path: Path, ra: float, dec: float, band: str) -> None:
    """Warn when a cached lightcurve's archive query is older than CACHE_STALE_DAYS."""
    # Caches written before queried_at was recorded fall back to the file's mtime.
    queried = datetime.fromisoformat(lc.queried_at).timestamp() if lc.queried_at else path.stat().st_mtime
    age_days = (time.time() - queried) / 86400
    if age_days > CACHE_STALE_DAYS:
        warnings.warn(
            f"({ra:.5f}, {dec:.5f}) [{band}]: cached lightcurve is {age_days:.0f} days old and "
            "misses any newer epochs; pass force_recompute=True to refresh it.",
            stacklevel=4,
        )


def _serve_from_cache(
    cache: CacheConfig,
    ra: float,
    dec: float,
    bands: tuple[str, ...] | list[str],
    ck: str,
    retry_bits: int,
    force_recompute: bool,
    show_progress: bool,
    warn_stale: bool = True,
) -> tuple[dict[str, Lightcurve], dict[str, Lightcurve], dict[str, set[str]], list[str]]:
    """Sort ``bands`` into what the cache answers and what must be fetched from IRSA.

    Returns ``(lightcurves, cached, retry_ids, todo)``: bands served whole from the
    cache; cached bands with epochs to retry, with those epochs' image ids; and every
    band that needs the archive (the retry bands and the uncached or stale ones).
    ``show_progress`` reports the bands loaded from the cache; ``warn_stale`` warns
    about cached lightcurves older than CACHE_STALE_DAYS.
    """
    lightcurves: dict[str, Lightcurve] = {}
    cached: dict[str, Lightcurve] = {}
    retry_ids: dict[str, set[str]] = {}
    todo: list[str] = []
    for band in bands:
        lc_fpath = lightcurve_path(cache, ra, dec, band)
        if lc_fpath.exists() and not force_recompute:
            lc = Lightcurve.load(lc_fpath)
            if lc.cache_key == ck:
                ids = {r["image_id"] for r in lc._rows if int(r["flags"]) & retry_bits}
                if warn_stale:
                    _warn_if_stale(lc, lc_fpath, ra, dec, band)
                if not ids:
                    if show_progress:
                        tqdm.write(f"({ra:.3f}, {dec:.3f}) [{band}] loaded from cache")
                    lightcurves[band] = lc
                    continue
                cached[band], retry_ids[band] = lc, ids
            # otherwise stale cache (settings changed) — fall through and recompute
        todo.append(band)
    return lightcurves, cached, retry_ids, todo


# ── Per-epoch workers ────────────────────────────────────────────────────────


def _exposure_stem(row: pd.Series) -> str:
    """``filefracday_field_filtercode_ccdid_qid``: names one quadrant exposure's files.

    filefracday identifies the exposure uniquely; field/filter/CCD/quadrant pick the
    file within it.  (JD to 3 decimals is 86 s, and ZTF takes back-to-back exposures
    of a field ~40 s apart, so a JD-based name let two exposures overwrite each other.)
    """
    return (
        f"{int(row['filefracday'])}_{int(row['field'])}_{row['filtercode']}"
        f"_{int(row['ccdid'])}_{int(row['qid'])}"
    )


def _exposure_uses(per_band: dict[str, pd.DataFrame]) -> Counter[str]:
    """How many epochs of one target's metadata fall on each quadrant exposure."""
    uses: Counter[str] = Counter()
    for df in per_band.values():
        if df.empty:
            continue
        # The same name _exposure_stem gives each row, built column-wise.
        stems = (
            df["filefracday"].astype(int).astype(str)
            + "_"
            + df["field"].astype(int).astype(str)
            + "_"
            + df["filtercode"].astype(str)
            + "_"
            + df["ccdid"].astype(int).astype(str)
            + "_"
            + df["qid"].astype(int).astype(str)
        )
        uses.update(stems.tolist())
    return uses


class _SidecarStore:
    """PSF sidecars shared by every source in a batch: one download per quadrant exposure.

    A sidecar describes the PSF of one quadrant exposure, so every target on that
    exposure can use the same file.  ``expected`` counts the targets whose metadata
    lists each exposure; each releases the exposure once it is done with it, and the
    file is deleted when the last one has, so the batch never holds more sidecars than
    its targets still need.
    """

    def __init__(self, root: Path, expected: Counter[str]) -> None:
        self.root = root
        self._pending = Counter(expected)
        self._lock = Lock()
        self._download_locks: dict[str, Lock] = {}
        self._failed: dict[str, Exception] = {}

    def fetch(self, stem: str, url: str, config: ZTForceConfig) -> Path:
        """The sidecar file for ``stem``, downloaded from ``url`` on first use.

        A download that failed is not retried for later targets on the exposure: the
        same error is raised again, so one dead file costs one round of retries.
        """
        path = self.root / f"{stem}.psf"
        with self._lock:
            lock = self._download_locks.setdefault(stem, Lock())
        with lock:
            if stem in self._failed:
                raise self._failed[stem]
            if not path.exists():
                try:
                    download_psf_sidecar(url, path, config)
                except Exception as exc:
                    self._failed[stem] = exc
                    raise
        return path

    def release(self, stem: str, n: int = 1) -> None:
        """Record that ``n`` of the expected users of ``stem`` no longer need it.

        The file is deleted once nobody is left, unless its download is still running
        (a target that exits early cannot stop one): it is then left to the batch
        directory's cleanup.
        """
        with self._lock:
            self._pending[stem] -= n
            if self._pending[stem] > 0:
                return
            del self._pending[stem]
            lock = self._download_locks.pop(stem, None)
        if lock is None or lock.acquire(blocking=False):
            (self.root / f"{stem}.psf").unlink(missing_ok=True)
            if lock is not None:
                lock.release()


def _release_sidecars(sidecars: _SidecarStore | None, pending: Counter[str]) -> None:
    """Release every use of the shared sidecars this target still holds."""
    if sidecars is None:
        return
    for stem, n in pending.items():
        if n > 0:
            sidecars.release(stem, n)
    pending.clear()


def _download_epoch(
    row: pd.Series,
    tmp_dir: Path,
    ra: float,
    dec: float,
    config: ZTForceConfig,
    sidecars: _SidecarStore | None = None,
) -> tuple[pd.Series, Path, Path]:
    """Download the FITS cutout and PSF sidecar for one epoch.

    The sidecar comes from ``sidecars`` when a batch shares them, so it is downloaded
    once for every target on the exposure.  Raises on failure so the caller can skip
    this epoch.
    """
    stem = _exposure_stem(row)
    local_fits = tmp_dir / f"{stem}.fits"
    fits_url = build_sci_url(row, ra, dec, suffix="sciimg.fits", cutout_size_arcmin=config.cutout_size_arcmin)
    psf_url = build_sci_url(row, ra, dec, suffix="sciimgdao.psf")
    download_fits(fits_url, local_fits, config)
    if sidecars is None:
        local_psf = tmp_dir / f"{stem}.psf"
        download_psf_sidecar(psf_url, local_psf, config)
    else:
        local_psf = sidecars.fetch(stem, psf_url, config)
    return row, local_fits, local_psf


def _process_one_epoch(
    fits_fpath: str,
    psf_fpath: str,
    ra: float,
    dec: float,
    band: str,
    image_id: str,
    config: ZTForceConfig,
    full_crpix: tuple[float, float] | None = None,
) -> dict:
    """Run forced PSF photometry for one epoch. Returns a result dict.

    ``full_crpix`` is the CRPIX of the full quadrant the cutout came from (archive
    metadata), needed to evaluate the spatially varying PSF at the right place.
    """
    try:
        img = ZTFImage(fits_fpath, band, config, full_crpix=full_crpix)
        parsed_psf = parse_daophot_psf(psf_fpath)
        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        result = forced_phot_at_position(img, parsed_psf, coord, core_taper=config.psf_fit_taper)
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


def _metadata_flags(row: pd.Series) -> int:
    """Quality-cut bits decidable from the archive metadata alone, before download.

    The metadata ``infobits`` is the authoritative one: bit 25 (bad photometric
    calibration) is set only in the archive database, never in the FITS header.
    """
    flags = 0
    infobits = row.get("infobits")
    if infobits is not None and np.isfinite(infobits) and infobits >= BAD_CALIBRATION_INFOBITS:
        flags |= FLAG_BAD_CALIBRATION
    seeing = row.get("seeing")  # arcsec
    if seeing is not None and np.isfinite(seeing) and seeing > MAX_SEEING_ARCSEC:
        flags |= FLAG_BAD_SEEING
    return flags


def _group(row: pd.Series) -> dict[str, int]:
    """The ZTF field / CCD / quadrant of a metadata row."""
    return dict(field=int(row["field"]), ccdid=int(row["ccdid"]), qid=int(row["qid"]))


def _image_id(row: pd.Series) -> str:
    """``field-ccdid-qid-obsjd``, with obsjd to 1e-5 d (0.9 s) so back-to-back exposures differ."""
    return f"{int(row['field'])}-{int(row['ccdid'])}-{int(row['qid'])}-{float(row['obsjd']):.5f}"


def _unmeasured_result(row: pd.Series, band: str, flags: int) -> dict:
    """A flagged epoch with no measurement (cut, unavailable, or failed), kept as a row."""
    nan = float("nan")
    infobits = row.get("infobits")
    return dict(
        flux=nan,
        flux_err=nan,
        mag=nan,
        mag_err=nan,
        chisq=nan,
        flags=flags,
        x_fit=nan,
        y_fit=nan,
        obsjd=float(row["obsjd"]),
        zero_point=nan,
        mag_limit=row.get("maglimit"),
        image_id=_image_id(row),
        band=band,
        infobits=int(infobits) if infobits is not None and np.isfinite(infobits) else None,
        seeing=row.get("seeing"),
        scisigpix=nan,
        **_group(row),
    )


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
    measure_flagged: bool = False,
    retry_unavailable: bool = False,
    _tqdm_position: int = 0,
    _tqdm_leave: bool = True,
    _download_executor: ThreadPoolExecutor | None = None,
    _metadata: dict[str, pd.DataFrame] | None = None,
    _sidecars: _SidecarStore | None = None,
    _cache_state: tuple[dict[str, Lightcurve], dict[str, Lightcurve], dict[str, set[str]], list[str]]
    | None = None,
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
        measure_flagged: Epochs the archive metadata already marks as failing the
            quality cuts (bad calibration, seeing > 4") are never detections or
            stacked either way.  If ``False`` (default) they are not downloaded and
            appear as flagged rows with NaN flux; if ``True`` they are downloaded and
            measured too, for inspection.
        retry_unavailable: Epochs whose files IRSA does not serve (``FLAG_UNAVAILABLE``)
            are not re-requested on later runs unless this is ``True``.  Epochs whose
            download failed transiently (``FLAG_DOWNLOAD_FAILED``) always are, reusing
            the rest of the cached lightcurve.

    Every epoch in the archive metadata appears as a row; ones that could not be
    measured carry a flag and NaN flux.  A warning summarises unavailable and failed
    epochs.

    Returns:
        Dict mapping band label (``"g"``, ``"r"``, ``"i"``) to a
        :class:`~ztforce.Lightcurve`.  Bands with no available images are
        omitted, so empty sky gives the cached bands only (often ``{}``).

    Raises:
        MetadataQueryError: The IRSA metadata search failed on every attempt.  An
            outage is not reported as empty sky; nothing is written to the cache.
    """
    cache = make_cache(data_dir)
    if config is None:
        config = build_config()

    ck = _cache_key(config, max_epochs, measure_flagged)

    # Serve what the cache can; everything else is fetched together below.  A cached
    # band with epochs to retry is reused and only those epochs are fetched again.
    retry_bits = FLAG_DOWNLOAD_FAILED | (FLAG_UNAVAILABLE if retry_unavailable else 0)
    # With a batch's shared sidecars, this target counts as a user of every exposure in
    # its metadata and releases each one when done with it (see _SidecarStore), on
    # every way out of this function.
    pending_sidecars: Counter[str] = Counter()
    if _metadata is not None and _sidecars is not None:
        pending_sidecars = _exposure_uses(_metadata)
    # The batch wrapper has already checked the cache for its targets.
    lightcurves, cached, retry_ids, todo = _cache_state or _serve_from_cache(
        cache, ra, dec, bands, ck, retry_bits, force_recompute, show_progress
    )
    if not todo:
        _release_sidecars(_sidecars, pending_sidecars)
        return lightcurves

    # One metadata query for every band still needed (IRSA's cost is the spatial
    # search, so this is ~1/3 the time of querying band by band), unless the batch
    # wrapper already fetched this position's metadata in its multi-position query.
    if _metadata is not None:
        metadata = {band: df for band, df in _metadata.items() if band in todo}
    else:
        # A MetadataQueryError (outage) propagates: only empty sky means "no images".
        # Nothing needs undoing: no shared sidecars are held without _metadata.
        try:
            metadata = query_sci_metadata_bands(ra, dec, todo, config)
        except NoImagesFoundError:
            metadata = {}
    if not metadata:
        _release_sidecars(_sidecars, pending_sidecars)
        return {**lightcurves, **cached}
    queried_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if max_epochs is not None:
        metadata = {band: df.tail(max_epochs).reset_index(drop=True) for band, df in metadata.items()}
    for band, ids in retry_ids.items():
        if band not in metadata:
            lightcurves[band] = cached.pop(band)
            continue
        df = metadata[band]
        metadata[band] = df[[_image_id(row) in ids for _, row in df.iterrows()]].reset_index(drop=True)

    # Quality cuts the metadata can decide.  Skipped epochs become flagged NaN rows.
    pre_flags: dict[str, int] = {}
    skipped: dict[str, list[dict]] = {band: [] for band in metadata}
    to_download: dict[str, pd.DataFrame] = {}
    for band, df in metadata.items():
        keep = []
        for _, row in df.iterrows():
            flags = _metadata_flags(row)
            pre_flags[_image_id(row)] = flags
            if flags and not measure_flagged:
                skipped[band].append(_unmeasured_result(row, band, flags))
            else:
                keep.append(row)
        to_download[band] = pd.DataFrame(keep, columns=df.columns)

    # Exposures this target will not download after all (cached bands, cut epochs,
    # epochs outside a retry) are released now, not when the target finishes.
    if _sidecars is not None:
        needed = _exposure_uses(to_download)
        _release_sidecars(_sidecars, pending_sidecars - needed)
        pending_sidecars = needed

    # Use a shared executor supplied by the batch wrapper, or own one locally.
    _own_executor = _download_executor is None
    dl_exec = _download_executor or ThreadPoolExecutor(max_workers=download_workers)

    desc_base = f"({ra:.3f}, {dec:.3f})"
    bar = tqdm(
        total=2 * sum(len(df) for df in to_download.values()),
        desc=f"{desc_base} downloading",
        position=_tqdm_position,
        leave=_tqdm_leave,
        disable=not show_progress,
        unit="step",
    )
    futures_by_band: dict[str, dict[Future, pd.Series]] = {}
    unmeasured_counts = {"unavailable": 0, "failed": 0, "processing error": 0}

    try:
        with tempfile.TemporaryDirectory() as _tmp:
            tmp_dir = Path(_tmp)

            # Download phase: every epoch of every band is submitted up front, so the
            # pool never idles between bands.  The pool is FIFO, so bands finish in
            # order and each is fitted and cached while the next is still downloading.
            for band, df in to_download.items():
                futures_by_band[band] = {
                    dl_exec.submit(_download_epoch, row, tmp_dir, ra, dec, config, _sidecars): row
                    for _, row in df.iterrows()
                }

            for band, futures in futures_by_band.items():
                image_triples: list[tuple[pd.Series, Path, Path]] = []
                results = list(skipped[band])
                for fut in as_completed(futures):
                    row = futures[fut]
                    try:
                        image_triples.append(fut.result())
                    except ProductUnavailableError:
                        flags = FLAG_UNAVAILABLE | pre_flags.get(_image_id(row), 0)
                        results.append(_unmeasured_result(row, band, flags))
                        unmeasured_counts["unavailable"] += 1
                        bar.update(1)
                    except Exception:
                        flags = FLAG_DOWNLOAD_FAILED | pre_flags.get(_image_id(row), 0)
                        results.append(_unmeasured_result(row, band, flags))
                        unmeasured_counts["failed"] += 1
                        bar.update(1)
                    bar.update(1)

                if not results and not image_triples and band not in cached:
                    continue

                # PSF photometry phase: sequential (CPU-fast, ~15 ms/epoch)
                bar.set_description(f"{desc_base} [{band}] fitting PSF")
                for row, fits_p, psf_p in image_triples:
                    image_id = _image_id(row)
                    full_crpix = (
                        (float(row["crpix1"]), float(row["crpix2"])) if "crpix1" in row.index else None
                    )
                    res = _process_one_epoch(
                        str(fits_p), str(psf_p), ra, dec, band, image_id, config, full_crpix
                    )
                    # The header lacks the archive-only bad-calibration bit: apply the
                    # metadata cuts and record the metadata infobits.
                    res["flags"] |= pre_flags.get(image_id, 0)
                    infobits = row.get("infobits")
                    if infobits is not None and np.isfinite(infobits):
                        res["infobits"] = int(infobits)
                    res.update(_group(row))
                    if not np.isfinite(res.get("obsjd", np.nan)):
                        # The image could not be read: keep the epoch, dated from metadata.
                        res["obsjd"] = float(row["obsjd"])
                        res.setdefault("seeing", row.get("seeing"))
                        unmeasured_counts["processing error"] += 1
                    results.append(res)
                    # Done with this epoch; don't hold every band's images on disk at once.
                    fits_p.unlink(missing_ok=True)
                    if _sidecars is None:
                        psf_p.unlink(missing_ok=True)
                    else:
                        stem = _exposure_stem(row)
                        _sidecars.release(stem)
                        pending_sidecars[stem] -= 1
                    bar.update(1)
                bar.set_description(f"{desc_base} downloading")

                results.sort(key=lambda d: d.get("obsjd", 0))

                # Assemble lightcurve: cached rows kept as they were, plus this run's.
                lc = Lightcurve(ra=ra, dec=dec)
                if band in cached:
                    lc._rows = [r for r in cached[band]._rows if r["image_id"] not in retry_ids[band]]
                for res in results:
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
                        field=res.get("field"),
                        ccdid=res.get("ccdid"),
                        qid=res.get("qid"),
                    )

                lc.cache_key = ck
                # A retry run adds no new epochs, so the cached query time still holds.
                lc.queried_at = cached[band].queried_at if band in cached else queried_at
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
        # Exposures this target never fitted (cut, failed, or an early exit) are released
        # too, so the shared sidecars are not kept for a user that will not come.
        _release_sidecars(_sidecars, pending_sidecars)

    if any(unmeasured_counts.values()):
        detail = ", ".join(f"{n} {what}" for what, n in unmeasured_counts.items() if n)
        warnings.warn(
            f"({ra:.5f}, {dec:.5f}): epochs not measured: {detail}. They are kept as flagged "
            "rows; failed downloads are retried on the next run.",
            stacklevel=2,
        )
    return lightcurves


def run_forced_photometry_batch(
    targets: list[SkyCoord],
    bands: tuple[str, ...] | list[str] = ("g", "r", "i"),
    data_dir: str | Path | None = None,
    config: ZTForceConfig | None = None,
    n_workers: int = 4,
    download_workers: int = 8,
    show_progress: bool = True,
    measure_flagged: bool = False,
    retry_unavailable: bool = False,
) -> list[dict[str, Lightcurve] | MetadataQueryError]:
    """Run forced photometry for a list of SkyCoord targets in parallel.

    Each target is processed by a dedicated thread; results are returned in the
    same order as ``targets``.  A single shared download thread pool (capped at
    ``download_workers``) is used across all active source workers so that
    concurrency is bounded at one level only.

    The archive metadata for every target the cache cannot serve is fetched in one
    multi-position query up front, which is far cheaper than IRSA's spatial search
    once per target.  PSF sidecars are shared: a quadrant exposure's sidecar is
    downloaded once however many targets fall on it.  If part of that query fails on
    every attempt, its targets are searched one at a time instead.

    Args:
        targets: Sky positions to process.
        bands: ZTF bands to process.  Any subset of ``("g", "r", "i")``.
        data_dir: Root directory for the on-disk cache.
        config: Credentials and runtime settings.
        n_workers: Number of targets to process concurrently.
        download_workers: Total number of concurrent epoch downloads shared
            across all active source workers.
        show_progress: If ``True`` (default), display tqdm progress bars.
        measure_flagged: See :func:`run_forced_photometry`.
        retry_unavailable: See :func:`run_forced_photometry`.

    Returns:
        List of band → :class:`~ztforce.Lightcurve` dicts, one per target.  A target
        whose metadata search failed on every attempt (an IRSA outage, not empty sky)
        gets the :class:`~ztforce.MetadataQueryError` in its place instead, with a
        warning; the other targets still complete.
    """
    if config is None:
        config = build_config()

    # One metadata query for every target that needs the archive.
    cache = make_cache(data_dir)
    ck = _cache_key(config, None, measure_flagged)
    retry_bits = FLAG_DOWNLOAD_FAILED | (FLAG_UNAVAILABLE if retry_unavailable else 0)
    positions = [(float(c.ra.deg), float(c.dec.deg)) for c in targets]
    cache_states = [
        _serve_from_cache(cache, ra, dec, bands, ck, retry_bits, False, show_progress)
        for ra, dec in positions
    ]
    uncached = [i for i, state in enumerate(cache_states) if state[3]]
    metadata: dict[int, dict[str, pd.DataFrame]] = {}
    if uncached:
        if show_progress:
            tqdm.write(f"Querying IRSA for the exposures covering {len(uncached)} positions")
        try:
            found = query_sci_metadata_positions([positions[i] for i in uncached], bands, config)
        except NoImagesFoundError:
            found = [{} for _ in uncached]
        # A target whose part of the query failed (None) is left out: its worker runs
        # its own single-position search.
        metadata = {i: m for i, m in zip(uncached, found, strict=True) if m is not None}
    # How many targets list each quadrant exposure: the shared sidecar store deletes
    # a sidecar once that many have released it.
    sidecar_users: Counter[str] = Counter()
    for per_band in metadata.values():
        sidecar_users.update(_exposure_uses(per_band))

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

    # One shared download pool for all source workers combined, and one directory of
    # shared PSF sidecars.
    with (
        ThreadPoolExecutor(max_workers=download_workers) as dl_exec,
        tempfile.TemporaryDirectory() as sidecar_dir,
    ):
        sidecars = _SidecarStore(Path(sidecar_dir), sidecar_users)

        def _run_one(idx: int) -> dict[str, Lightcurve]:
            pos = _acquire_position()
            try:
                return run_forced_photometry(
                    ra=positions[idx][0],
                    dec=positions[idx][1],
                    bands=bands,
                    data_dir=data_dir,
                    config=config,
                    show_progress=show_progress,
                    measure_flagged=measure_flagged,
                    retry_unavailable=retry_unavailable,
                    _tqdm_position=pos,
                    _tqdm_leave=False,
                    _download_executor=dl_exec,
                    _metadata=metadata.get(idx),
                    _sidecars=sidecars if idx in metadata else None,
                    _cache_state=cache_states[idx],
                )
            finally:
                _release_position(pos)
                main_bar.update(1)

        results: list[dict[str, Lightcurve] | MetadataQueryError] = [{} for _ in range(len(targets))]
        with ThreadPoolExecutor(max_workers=n_workers) as src_exec:
            future_to_idx = {src_exec.submit(_run_one, i): i for i in range(len(targets))}
            for future in as_completed(future_to_idx):
                try:
                    results[future_to_idx[future]] = future.result()
                except MetadataQueryError as exc:
                    results[future_to_idx[future]] = exc

    main_bar.close()
    failed = [i for i, r in enumerate(results) if isinstance(r, MetadataQueryError)]
    if failed:
        warnings.warn(
            f"IRSA metadata search failed for {len(failed)} of {len(targets)} targets "
            f"(indices {failed}); their results are MetadataQueryError, retry them later.",
            stacklevel=2,
        )
    return results
