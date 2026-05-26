"""Orchestration: parallel download + forced PSF photometry."""

from __future__ import annotations

import hashlib
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from astropy.coordinates import SkyCoord

from .cache import _PHOTOMETRY_VERSION, lightcurve_path, make_cache
from .config import ZTForceConfig, build_config
from .exceptions import NoImagesFoundError
from .image import ZTFImage
from .lightcurve import Lightcurve
from .psf import forced_phot_at_position, parse_daophot_psf
from .ztf_images import build_sci_url, download_fits, download_psf_sidecar, query_sci_metadata

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


# ── Worker (must be importable at module level for ProcessPoolExecutor on macOS) ──


def _process_one_image(
    fits_fpath: str,
    psf_fpath: str,
    ra: float,
    dec: float,
    band: str,
    image_id: str,
    irsa_user: str,
    irsa_pass: str,
    default_gain: float,
) -> dict:
    """Run forced PSF photometry for one image. Returns a result dict."""
    from .config import ZTForceConfig

    config = ZTForceConfig(
        irsa_user=irsa_user,
        irsa_pass=irsa_pass,
        default_gain=default_gain,
    )
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
    except Exception:
        result = dict(
            flux=float("nan"),
            flux_err=float("nan"),
            mag=float("nan"),
            mag_err=float("nan"),
            flags=2,
            x_fit=float("nan"),
            y_fit=float("nan"),
            obsjd=float("nan"),
            zero_point=float("nan"),
            mag_limit=None,
            image_id=image_id,
            band=band,
        )
        import traceback

        traceback.print_exc()
    return result


def _worker_kwargs(config: ZTForceConfig) -> dict:
    """Extract picklable config fields for the worker function."""
    return dict(
        irsa_user=config.irsa_user,
        irsa_pass=config.irsa_pass,
        default_gain=config.default_gain,
    )


# ── Download phase (I/O-bound, threaded) ─────────────────────────────────────


def _download_all(
    df: pd.DataFrame,
    ra: float,
    dec: float,
    band: str,
    tmp_dir: Path,
    config: ZTForceConfig,
    n_workers: int,
) -> list[tuple[pd.Series, Path, Path]]:
    """Download all FITS cutouts + PSF sidecars in parallel into tmp_dir."""

    def _download_one(row):
        obsjd = float(row["obsjd"])
        stem = f"{int(row['field'])}-{int(row['ccdid'])}-{int(row['qid'])}-{obsjd:.3f}"
        local_fits = tmp_dir / f"{stem}.fits"
        local_psf = tmp_dir / f"{stem}.psf"
        fits_url = build_sci_url(
            row, ra, dec, suffix="sciimg.fits", cutout_size_arcmin=config.cutout_size_arcmin
        )
        psf_url = build_sci_url(row, ra, dec, suffix="sciimgdao.psf")
        try:
            download_fits(fits_url, local_fits, config)
            download_psf_sidecar(psf_url, local_psf, config)
            return row, local_fits, local_psf
        except Exception:
            return None

    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_download_one, row): i for i, (_, row) in enumerate(df.iterrows())}
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)
    # Re-sort by obsjd
    results.sort(key=lambda t: float(t[0]["obsjd"]))
    return results


# ── PSF photometry phase (CPU-bound, multiprocess) ────────────────────────────


def _run_psf_parallel(
    image_triples: list[tuple[pd.Series, Path, Path]],
    ra: float,
    dec: float,
    band: str,
    config: ZTForceConfig,
    n_workers: int,
) -> list[dict]:
    """Run forced PSF photometry on all images in parallel processes."""
    wkwargs = _worker_kwargs(config)
    args_list = [
        (
            str(fits_p),
            str(psf_p),
            ra,
            dec,
            band,
            f"{int(row['field'])}-{int(row['ccdid'])}-{int(row['qid'])}-{float(row['obsjd']):.3f}",
        )
        for row, fits_p, psf_p in image_triples
    ]

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_process_one_image, *args, **wkwargs) for args in args_list]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda d: d.get("obsjd", 0))
    return results


# ── Public API ────────────────────────────────────────────────────────────────


def run_forced_photometry(
    ra: float,
    dec: float,
    bands: tuple[str, ...] | list[str] = ("g", "r", "i"),
    data_dir: str | Path | None = None,
    config: ZTForceConfig | None = None,
    n_download_workers: int = 4,
    n_psf_workers: int = 4,
    max_epochs: int | None = None,
    force_recompute: bool = False,
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
        n_download_workers: Number of parallel threads for FITS downloads.
        n_psf_workers: Number of parallel processes for PSF photometry.
        max_epochs: If set, process only the *most recent* ``max_epochs``
            exposures per band.  Useful for quick tests.
        force_recompute: If ``True``, ignore any cached lightcurve and
            redownload + recompute from scratch, overwriting the cache.

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

    for band in bands:
        lc_fpath = lightcurve_path(cache, ra, dec, band)

        # Cache hit: load and return if the key matches
        if lc_fpath.exists() and not force_recompute:
            lc = Lightcurve.load(lc_fpath)
            if lc.cache_key == ck:
                lightcurves[band] = lc
                continue
            # stale cache (settings changed) — fall through and recompute

        # Query metadata
        try:
            df = query_sci_metadata(ra, dec, band, config)
        except NoImagesFoundError:
            continue

        if max_epochs is not None:
            df = df.tail(max_epochs).reset_index(drop=True)

        # Download to a temp dir (cleaned up after photometry)
        with tempfile.TemporaryDirectory() as _tmp:
            image_triples = _download_all(df, ra, dec, band, Path(_tmp), config, n_download_workers)
            if not image_triples:
                continue

            # PSF photometry phase (multiprocess) — runs while temp files still exist
            results = _run_psf_parallel(image_triples, ra, dec, band, config, n_psf_workers)

        # Assemble lightcurve
        lc = Lightcurve(ra=ra, dec=dec)
        for res in results:
            if not res.get("obsjd") or (res.get("obsjd") != res.get("obsjd")):
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
            )

        lc.cache_key = ck
        lc.save(lc_fpath)
        lightcurves[band] = lc

    return lightcurves


def run_forced_photometry_batch(
    targets: list[SkyCoord],
    bands: tuple[str, ...] | list[str] = ("g", "r", "i"),
    data_dir: str | Path | None = None,
    config: ZTForceConfig | None = None,
    n_download_workers: int = 4,
    n_psf_workers: int = 4,
) -> list[dict[str, Lightcurve]]:
    """Run forced photometry for a list of SkyCoord targets.

    Returns a list of band → Lightcurve dicts, one per target.
    All targets share cached FITS files when they fall on the same image.
    """
    if config is None:
        config = build_config()

    return [
        run_forced_photometry(
            ra=float(coord.ra.deg),
            dec=float(coord.dec.deg),
            bands=bands,
            data_dir=data_dir,
            config=config,
            n_download_workers=n_download_workers,
            n_psf_workers=n_psf_workers,
        )
        for coord in targets
    ]
