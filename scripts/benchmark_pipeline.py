"""
Benchmark and profile the ztforce forced-photometry pipeline step by step.

Runs :func:`ztforce.run_forced_photometry` on a set of sky positions and times every
stage of the pipeline by wrapping the functions it calls:

- ``metadata query``   IRSA spatial search for the exposures covering the position
- ``download epoch``   FITS cutout + PSF sidecar for one epoch (both files)
-   ``download fits``    the cutout alone (``download fits: validate`` is its FITS check)
-   ``download psf``     the sidecar alone
- ``process epoch``    everything from opening the cutout to the flux for one epoch
-   ``load fits``        reading the cutout into memory
-   ``parse psf``        parsing the DAOPhot sidecar
-   ``forced fit``       the PSF fit itself
- ``save lightcurve``  writing the per-band ECSV

It also reports how much the pipeline stores: the per-band lightcurve files written
to the cache, and what it would store if the FITS cutouts and PSF sidecars (temporary
today, deleted after fitting) were kept.

The report has two tables.  Per source: the metadata query, the download phase
(wall time from the first download to the last; downloads run ``--download-workers``
at a time), the fitting phase (the fits run one after another, so their sum), the
lightcurve save and the total wall time, each averaged over sources.  Per epoch: the
download and fit steps, with per-epoch statistics and the total per source.  For
downloads the per-epoch time is latency, not throughput, because they overlap.

Nothing cached is reused: lightcurves are written to a fresh temporary cache that is
deleted afterwards and ``force_recompute`` is set.  The sources are 20 real ZTF
objects (SOURCES below), four per field in five fields, spanning g = 13 to 21.

Needs IRSA credentials (see README).  Run from the repo root:

    python scripts/benchmark_pipeline.py                        # all 20 sources, 50 epochs/band
    python scripts/benchmark_pipeline.py --n-sources 5          # the first five only
    python scripts/benchmark_pipeline.py --max-epochs 0         # every epoch (slow)
    python scripts/benchmark_pipeline.py --profile bench.prof   # also cProfile the run
    python scripts/benchmark_pipeline.py --out bench.json       # keep the raw timings
"""

from __future__ import annotations

import argparse
import cProfile
import functools
import json
import pstats
import shutil
import statistics
import tempfile
import time
import warnings
from pathlib import Path
from threading import Lock

import numpy as np
import ztforce.image
import ztforce.lightcurve
import ztforce.pipeline
import ztforce.ztf_images
from ztforce import NoImagesFoundError, build_config, run_forced_photometry
from ztforce.cache import lightcurve_path, make_cache

# ZTF DR21 objects with > 100 good g-band epochs: one per magnitude bin (13-15, 15-17,
# 17-19, 19-21) within 6 arcmin of five positions (a dense field at (95.57, 77.50), the
# CVn and Praesepe regression-test fields, (210.08, -6.88), and the Lyra field at
# (280, 40)).  Pulled on 2026-11-06 with
#   Irsa.query_region(coord, catalog="ztf_objects_dr21", radius=6 * u.arcmin)
# from astroquery.ipac.irsa, picking a random object in each bin (numpy seed 0).
SOURCES: list[tuple[float, float]] = [
    (95.503967, 77.432706),  # g = 14.2, 406 good epochs
    (95.261577, 77.439945),  # g = 15.7, 405 good epochs
    (95.488952, 77.431081),  # g = 18.4, 570 good epochs
    (95.191813, 77.541767),  # g = 19.9, 438 good epochs
    (199.760382, 30.197774),  # g = 14.7, 694 good epochs
    (199.832889, 30.178744),  # g = 16.7, 247 good epochs
    (199.753774, 30.240686),  # g = 18.8, 688 good epochs
    (199.805124, 30.180503),  # g = 20.6, 338 good epochs
    (130.009426, 19.787016),  # g = 13.3, 270 good epochs
    (130.060725, 19.737900),  # g = 15.3, 322 good epochs
    (130.068990, 19.674766),  # g = 18.2, 300 good epochs
    (130.124694, 19.719541),  # g = 20.2, 238 good epochs
    (210.165153, -6.852346),  # g = 14.5, 197 good epochs
    (210.088630, -6.873383),  # g = 15.6, 197 good epochs
    (210.129785, -6.942791),  # g = 18.1, 195 good epochs
    (210.122016, -6.866088),  # g = 20.1, 129 good epochs
    (280.074893, 40.019931),  # g = 13.3, 1121 good epochs
    (279.941812, 39.943957),  # g = 15.1, 1123 good epochs
    (279.878082, 40.024749),  # g = 17.9, 1123 good epochs
    (280.048196, 39.938903),  # g = 20.2, 906 good epochs
]

# (module, attribute, label).  Each is replaced by a timed wrapper for the run.
# pipeline.py imports its helpers by name, so they are patched where it looks them up.
STEPS = [
    (ztforce.pipeline, "query_sci_metadata_bands", "metadata query"),
    (ztforce.pipeline, "_download_epoch", "download epoch"),
    (ztforce.pipeline, "download_fits", "  download fits"),
    (ztforce.ztf_images, "_validate_fits", "    download fits: validate"),
    (ztforce.pipeline, "download_psf_sidecar", "  download psf"),
    (ztforce.pipeline, "_process_one_epoch", "process epoch"),
    (ztforce.image.ZTFImage, "_load_fits", "  load fits"),
    (ztforce.pipeline, "parse_daophot_psf", "  parse psf"),
    (ztforce.pipeline, "forced_phot_at_position", "  forced fit"),
    (ztforce.lightcurve.Lightcurve, "save", "save lightcurve"),
]
STEP_LABELS = [label for _, _, label in STEPS]
# Download steps whose second argument is the destination file: its size is recorded.
FILE_STEPS = {"  download fits", "  download psf"}


class StepTimer:
    """Records (step, source, start, end) for every call of the wrapped functions."""

    def __init__(self) -> None:
        self.records: list[tuple[str, int, float, float]] = []
        self.file_sizes: list[tuple[str, int, int]] = []  # (step, source, bytes) of downloaded files
        self.source = -1
        self._lock = Lock()
        self._originals: list[tuple[object, str, object]] = []

    def install(self) -> None:
        """Replace every step function with a timed wrapper."""
        for owner, name, label in STEPS:
            original = getattr(owner, name)
            self._originals.append((owner, name, original))
            setattr(owner, name, self._wrap(original, label))

    def uninstall(self) -> None:
        """Restore the original functions."""
        for owner, name, original in self._originals:
            setattr(owner, name, original)
        self._originals.clear()

    def _wrap(self, func, label):
        @functools.wraps(func)
        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            finally:
                t1 = time.perf_counter()
                with self._lock:
                    self.records.append((label, self.source, t0, t1))
            if label in FILE_STEPS:
                size = Path(args[1]).stat().st_size
                with self._lock:
                    self.file_sizes.append((label, self.source, size))
            return result

        return timed

    def durations(self, label: str, source: int | None = None) -> list[float]:
        """Per-call durations of one step, optionally for one source only."""
        return [t1 - t0 for lab, src, t0, t1 in self.records if lab == label and source in (None, src)]

    def sizes(self, label: str, source: int | None = None) -> list[int]:
        """Sizes in bytes of the files one download step fetched, optionally for one source only."""
        return [n for lab, src, n in self.file_sizes if lab == label and source in (None, src)]

    def span(self, label: str, source: int) -> float:
        """Wall time from the first call's start to the last call's end, for one source."""
        times = [(t0, t1) for lab, src, t0, t1 in self.records if lab == label and src == source]
        if not times:
            return 0.0
        return max(t1 for _, t1 in times) - min(t0 for t0, _ in times)


def _stats(values: list[float]) -> dict:
    """mean / median / p90 / max of a list of seconds."""
    if not values:
        return dict(n=0, mean_s=float("nan"), median_s=float("nan"), p90_s=float("nan"), max_s=float("nan"))
    return dict(
        n=len(values),
        mean_s=statistics.fmean(values),
        median_s=statistics.median(values),
        p90_s=float(np.percentile(values, 90)),
        max_s=max(values),
    )


PER_EPOCH_STEPS = [label for label in STEP_LABELS if label not in ("metadata query", "save lightcurve")]


def summarize(timer: StepTimer, sources: list[dict]) -> dict:
    """Per-source phase statistics and per-epoch step statistics."""
    done = [s["index"] for s in sources if s["status"] == "ok"]
    per_source = {
        "metadata query": [sum(timer.durations("metadata query", i)) for i in done],
        "download phase": [timer.span("download epoch", i) for i in done],
        "fit phase": [sum(timer.durations("process epoch", i)) for i in done],
        "save lightcurve": [sum(timer.durations("save lightcurve", i)) for i in done],
        "total wall": [s["wall_s"] for s in sources if s["status"] == "ok"],
    }
    per_epoch = {}
    for label in PER_EPOCH_STEPS:
        d = timer.durations(label)
        if d:
            per_epoch[label] = {
                **_stats(d),
                "per_source_total_s": statistics.fmean(sum(timer.durations(label, i)) for i in done),
            }
    images = {
        "fits_cutout_bytes": _stats(timer.sizes("  download fits")),
        "psf_sidecar_bytes": _stats(timer.sizes("  download psf")),
        "per_source_bytes": _stats([s["image_bytes"] for s in sources if s["status"] == "ok"]),
    }
    return {
        "per_source": {k: _stats(v) for k, v in per_source.items() if v},
        "per_epoch": per_epoch,
        "images": images,
    }


def print_report(sources: list[dict], summary: dict) -> None:
    """Print the source list, the per-source phase table and the per-epoch step table."""
    print("\nSources")
    print(f"{'ra':>10} {'dec':>9} {'epochs':>7} {'measured':>9} {'wall s':>8} {'stored kB':>10}  status")
    for s in sources:
        print(
            f"{s['ra']:10.5f} {s['dec']:9.5f} {s['n_epochs']:7d} {s['n_measured']:9d} "
            f"{s['wall_s']:8.1f} {s['stored_bytes'] / 1e3:10.1f}  {s['status']}"
        )
    done = [s for s in sources if s["status"] == "ok"]
    if not done:
        print("\nNo source completed; nothing to summarize.")
        return
    epochs = [s["n_epochs"] for s in done]
    measured = [s["n_measured"] for s in done]
    print(
        f"\n{len(done)} sources completed, {statistics.fmean(epochs):.0f} epochs each on average, "
        f"of which {statistics.fmean(measured):.0f} were downloaded and fitted (the rest failed the "
        "metadata quality cuts or could not be downloaded)."
    )

    print("\nPer source  (seconds per source; statistics over sources)")
    hdr = f"{'phase':<28} {'mean':>8} {'median':>8} {'p90':>8} {'max':>8}"
    print(hdr)
    print("-" * len(hdr))
    for label, st in summary["per_source"].items():
        print(f"{label:<28} {st['mean_s']:8.2f} {st['median_s']:8.2f} {st['p90_s']:8.2f} {st['max_s']:8.2f}")
    print(
        "download phase: first download start to last download end (downloads overlap).\n"
        "fit phase: sum of the sequential per-epoch fits."
    )

    print("\nPer epoch  (seconds per epoch; statistics over epochs, plus the summed time per source)")
    hdr = f"{'step':<28} {'epochs':>7} {'mean':>8} {'median':>8} {'p90':>8} {'max':>8} {'per source':>11}"
    print(hdr)
    print("-" * len(hdr))
    for label, st in summary["per_epoch"].items():
        print(
            f"{label:<28} {st['n']:>7d} {st['mean_s']:8.3f} {st['median_s']:8.3f} "
            f"{st['p90_s']:8.3f} {st['max_s']:8.3f} {st['per_source_total_s']:11.2f}"
        )
    print(
        "Indented steps are parts of the step above them.  Download times are per-epoch latency; "
        "several run at once,\nso their per-source sum exceeds the download phase wall time above."
    )

    stored = [s["stored_bytes"] for s in done]
    per_epoch_bytes = sum(stored) / max(sum(epochs), 1)
    print(
        f"\nStored  (lightcurve files written to the cache; cutouts are deleted after fitting)\n"
        f"{sum(stored) / 1e3:.1f} kB in total, {statistics.fmean(stored) / 1e3:.1f} kB per source on average "
        f"({statistics.fmean(epochs):.0f} epochs each), {per_epoch_bytes:.0f} bytes per epoch row."
    )
    fits_mb = summary["images"]["fits_cutout_bytes"]["mean_s"] / 1e6
    psf_mb = summary["images"]["psf_sidecar_bytes"]["mean_s"] / 1e6
    img_src = summary["images"]["per_source_bytes"]
    n_files = summary["images"]["fits_cutout_bytes"]["n"]
    print(
        f"\nIf the images were kept  (every FITS cutout and PSF sidecar downloaded in this run)\n"
        f"{sum(s['image_bytes'] for s in done) / 1e6:.1f} MB in total for {n_files} epochs: "
        f"{(fits_mb + psf_mb) * 1e3:.0f} kB per epoch (cutout {fits_mb * 1e3:.0f} kB + sidecar "
        f"{psf_mb * 1e3:.0f} kB), {img_src['mean_s'] / 1e6:.1f} MB per source on average "
        f"({statistics.fmean(measured):.0f} measured epochs each; max {img_src['max_s'] / 1e6:.1f} MB)."
    )


def main() -> None:
    """Parse arguments, run the benchmark, print and optionally save the results."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--n-sources", type=int, default=len(SOURCES), help=f"use the first N of the {len(SOURCES)} sources"
    )
    ap.add_argument(
        "--max-epochs", type=int, default=50, help="most recent epochs per band; 0 for all (default 50)"
    )
    ap.add_argument("--bands", nargs="+", default=["g", "r", "i"], help="bands to process (default g r i)")
    ap.add_argument("--download-workers", type=int, default=8, help="concurrent downloads (default 8)")
    ap.add_argument("--out", type=Path, default=None, help="write the summary and raw timings as JSON")
    ap.add_argument("--profile", type=Path, default=None, help="cProfile the run and write stats here")
    ap.add_argument("--quiet", action="store_true", help="hide the pipeline's progress bars")
    args = ap.parse_args()

    config = build_config()
    max_epochs = args.max_epochs or None
    positions = SOURCES[: args.n_sources]
    # A fresh, empty cache so no lightcurve is ever read from disk.
    cache_dir = Path(tempfile.mkdtemp(prefix="ztforce_bench_"))

    timer = StepTimer()
    timer.install()
    profiler = cProfile.Profile() if args.profile else None
    sources: list[dict] = []
    print(
        f"{len(positions)} sources, bands {'/'.join(args.bands)}, "
        f"{'all' if max_epochs is None else max_epochs} epochs per band, "
        f"{args.download_workers} download workers, cache at {cache_dir}"
    )
    try:
        if profiler:
            profiler.enable()
        for i, (ra, dec) in enumerate(positions):
            timer.source = i
            rec = dict(
                index=i,
                ra=ra,
                dec=dec,
                n_epochs=0,
                n_measured=0,
                wall_s=0.0,
                stored_bytes=0,
                image_bytes=0,
                status="ok",
            )
            t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # unmeasured-epoch summaries are not timing info
                    lcs = run_forced_photometry(
                        ra,
                        dec,
                        bands=args.bands,
                        data_dir=cache_dir,
                        config=config,
                        max_epochs=max_epochs,
                        force_recompute=True,
                        show_progress=not args.quiet,
                        download_workers=args.download_workers,
                    )
            except NoImagesFoundError:
                rec["status"] = "no images"
            except Exception as exc:  # keep going; the failure is part of the report
                rec["status"] = f"error: {type(exc).__name__}: {exc}"
            else:
                rec["n_epochs"] = sum(len(lc) for lc in lcs.values())
                rec["n_measured"] = len(timer.durations("process epoch", i))
                lc_dir = lightcurve_path(make_cache(cache_dir), ra, dec, args.bands[0]).parent
                rec["stored_bytes"] = sum(f.stat().st_size for f in lc_dir.iterdir() if f.is_file())
                rec["image_bytes"] = sum(sum(timer.sizes(step, i)) for step in FILE_STEPS)
            rec["wall_s"] = time.perf_counter() - t0
            sources.append(rec)
            print(f"[{i + 1}/{len(positions)}] ({ra:.5f}, {dec:.5f}): {rec['wall_s']:.1f} s, {rec['status']}")
    except KeyboardInterrupt:
        print("\nInterrupted; reporting the sources finished so far.")
    finally:
        if profiler:
            profiler.disable()
        timer.uninstall()
        shutil.rmtree(cache_dir, ignore_errors=True)

    summary = summarize(timer, sources)
    print_report(sources, summary)

    if profiler:
        profiler.dump_stats(str(args.profile))
        print(f"\ncProfile stats written to {args.profile} (main thread only: fits and orchestration,")
        print("not the download threads).  Top functions by cumulative time:\n")
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(25)

    if args.out:
        payload = {
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "sources": sources,
            **summary,
            "file_sizes": [
                {"step": lab.strip(), "source": src, "bytes": n} for lab, src, n in timer.file_sizes
            ],
            "records": [
                {"step": lab, "source": src, "start": t0, "end": t1} for lab, src, t0, t1 in timer.records
            ],
        }
        args.out.write_text(json.dumps(payload, indent=1))
        print(f"\nRaw timings written to {args.out}")


if __name__ == "__main__":
    main()
