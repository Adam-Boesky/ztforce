"""
Build the frozen test fixture for ``tests/ztforce/integration/test_psfcat_agreement.py``.

The test checks ztforce's forced photometry against the ZTF pipeline's own
PSF-fit catalog (``psfcat``) for the same science image.  Both use the same
pixels, the same DAOPhot PSF sidecar and the same MAGZP, so zero points,
colour terms and calibration cancel: a correct forced fit on an isolated,
unsaturated star should reproduce the catalog flux to ~1%.

For each selected image this downloads the PSF sidecar and the full-quadrant
psfcat, picks isolated unsaturated stars spread over the quadrant (one per
cell of a 3x3 grid) plus blank-sky positions, and saves a small IBE cutout
around each.  Everything the test needs is written under
``tests/data/psfcat_agreement/`` with a ``manifest.json``; the psfcat itself
is not kept.

Needs IRSA credentials (see README).  Run from the repo root:

    python scripts/build_psfcat_fixture.py
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from ztforce import build_config
from ztforce.ztf_images import (
    build_sci_url,
    download_fits,
    download_psf_sidecar,
    query_sci_metadata_bands,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "data" / "psfcat_agreement"

# A dense, high-cadence position covered by several field/CCD/quadrant combinations.
RA, DEC = 95.56940, 77.50087
BANDS = ("g", "r")
MIN_JD = 2459500.0  # IBE cutouts of older epochs have returned 404s
N_QUADRANTS = 3  # field/ccdid/qid combinations to use
CUTOUT_ARCMIN = 1.2

# Star selection (psfcat pixel units).  By flux in DN, not psfcat ``snr``: psfcat's
# ``sigflux`` includes a per-image systematic floor, so its S/N is capped (at ~50 in
# some images) and is not a usable statistical S/N.
FLUX_RANGE_DN = (3000.0, 80000.0)  # well above sky noise, below saturation for typical seeing
FLUX_TARGET_DN = 15000.0
EDGE_PX = 60  # stay clear of quadrant edges so cutouts are never clipped
ISOLATION_PX = 15  # no catalogued neighbour at all within this radius
BRIGHT_NEIGHBOUR_PX = 30  # ...and none brighter than 10% of the star within this one
MAX_PEAK_FRAC_SATURATE = 0.5
BLANK_CLEAR_PX = 25  # blank-sky positions: nothing catalogued within this radius
N_BLANK = 4
QUAD_NX, QUAD_NY = 3072, 3080


def _stem(band: str, row) -> str:
    return f"{band}_{int(row['field'])}_{int(row['ccdid'])}_{int(row['qid'])}_{float(row['obsjd']):.4f}"


def _pick_images(metadata: dict) -> list[tuple[str, object]]:
    """Two epochs (best and worst seeing) per band for the most-observed quadrants."""
    picks = []
    for band in BANDS:
        df = metadata[band]
        df = df[df["obsjd"] > MIN_JD]
        df = df[df["infobits"] < 2**25]
        groups = df.groupby(["field", "ccdid", "qid"]).size().sort_values(ascending=False)
        for key in groups.index[:N_QUADRANTS]:
            sub = df[(df["field"] == key[0]) & (df["ccdid"] == key[1]) & (df["qid"] == key[2])]
            sub = sub.sort_values("seeing")
            k = len(sub) // 10  # 10th and 90th percentile seeing, not the extremes
            for row in (sub.iloc[k], sub.iloc[len(sub) - 1 - k]):
                picks.append((band, row))
    return picks


def _star_candidates(cat) -> list[list[int]]:
    """Per 3x3 quadrant cell, isolated stars ordered by flux closeness to FLUX_TARGET_DN.

    Mid-flux rather than brightest, so the first choice is rarely near saturation.
    """
    x, y, flux = cat["xpos"], cat["ypos"], cat["flux"]
    ok = (
        (cat["flags"] == 0)
        & (flux > FLUX_RANGE_DN[0])
        & (flux < FLUX_RANGE_DN[1])
        & (x > EDGE_PX)
        & (x < QUAD_NX - EDGE_PX)
        & (y > EDGE_PX)
        & (y < QUAD_NY - EDGE_PX)
    )
    cells = []
    for gx in range(3):
        for gy in range(3):
            cell = np.where(ok & (x // (QUAD_NX / 3) == gx) & (y // (QUAD_NY / 3) == gy))[0]
            cell = cell[np.argsort(np.abs(np.log(flux[cell] / FLUX_TARGET_DN)))]
            isolated = []
            for i in cell:
                d = np.hypot(x - x[i], y - y[i])
                d[i] = np.inf
                if d.min() < ISOLATION_PX or np.any((d < BRIGHT_NEIGHBOUR_PX) & (flux > 0.1 * flux[i])):
                    continue
                isolated.append(int(i))
                if len(isolated) == 3:
                    break
            cells.append(isolated)
    return cells


def _saturated(path: Path) -> bool:
    with fits.open(path) as hdul:
        data, hdr = hdul[0].data, hdul[0].header
        c = np.array(data.shape) // 2
        peak = np.nanmax(data[c[0] - 3 : c[0] + 4, c[1] - 3 : c[1] + 4])
        return bool(peak > MAX_PEAK_FRAC_SATURATE * float(hdr["SATURATE"]))


def _select_blanks(cat, rng) -> list[tuple[float, float]]:
    x, y = cat["xpos"], cat["ypos"]
    blanks = []
    for _ in range(2000):
        bx = rng.uniform(EDGE_PX, QUAD_NX - EDGE_PX)
        by = rng.uniform(EDGE_PX, QUAD_NY - EDGE_PX)
        if np.hypot(x - bx, y - by).min() > BLANK_CLEAR_PX:
            blanks.append((bx, by))
        if len(blanks) == N_BLANK:
            break
    return blanks


def main() -> None:
    """Download, select, and write the fixture."""
    cfg = build_config()
    metadata = query_sci_metadata_bands(RA, DEC, BANDS, cfg)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    rng = np.random.default_rng(0)
    manifest = {"images": []}

    for band, row in _pick_images(metadata):
        stem = _stem(band, row)
        img_dir = OUT_DIR / stem
        img_dir.mkdir()
        try:
            download_psf_sidecar(
                build_sci_url(row, RA, DEC, suffix="sciimgdao.psf"), img_dir / "psf.psf", cfg
            )
            with tempfile.TemporaryDirectory() as tmp:
                cat_path = Path(tmp) / "psfcat.fits"
                download_psf_sidecar(build_sci_url(row, RA, DEC, suffix="psfcat.fits"), cat_path, cfg)
                with fits.open(cat_path) as hdul:
                    cat = hdul[1].data.copy()
        except Exception as exc:  # noqa: BLE001
            print(f"{stem}: skipped ({exc})")
            shutil.rmtree(img_dir)
            continue

        def _cutout(name: str, ra: float, dec: float, row=row, img_dir=img_dir) -> Path | None:
            path = img_dir / name
            url = build_sci_url(row, ra, dec, "sciimg.fits", cutout_size_arcmin=CUTOUT_ARCMIN)
            try:
                download_fits(url, path, cfg)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {name}: {exc}")
                return None
            return path

        kept = []
        for cell in _star_candidates(cat):
            for i in cell:
                name = f"star_{len(kept):02d}.fits"
                path = _cutout(name, float(cat["ra"][i]), float(cat["dec"][i]))
                if path is None:
                    continue
                if _saturated(path):
                    path.unlink()
                    continue
                kept.append(
                    dict(
                        kind="star",
                        file=name,
                        ra=float(cat["ra"][i]),
                        dec=float(cat["dec"][i]),
                        xpos=float(cat["xpos"][i]),
                        ypos=float(cat["ypos"][i]),
                        flux=float(cat["flux"][i]),
                        sigflux=float(cat["sigflux"][i]),
                        snr=float(cat["snr"][i]),
                    )
                )
                break

        # Blank positions are chosen in quadrant pixels; map them to the sky with a
        # quadratic fit to the catalog's own (xpos, ypos) -> (ra, dec), which is
        # good to well under the 25 px clearance.
        def _design(px, py):
            return np.column_stack([np.ones_like(px), px, py, px * px, px * py, py * py])

        design = _design(cat["xpos"].astype(float), cat["ypos"].astype(float))
        cra, *_ = np.linalg.lstsq(design, cat["ra"], rcond=None)
        cdec, *_ = np.linalg.lstsq(design, cat["dec"], rcond=None)
        for k, (bx, by) in enumerate(_select_blanks(cat, rng)):
            v = _design(np.array([bx]), np.array([by]))[0]
            ra, dec = float(v @ cra), float(v @ cdec)
            name = f"blank_{k:02d}.fits"
            if _cutout(name, ra, dec) is not None:
                kept.append(dict(kind="blank", file=name, ra=ra, dec=dec, xpos=bx, ypos=by))

        meta_row = {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in row.items()
            if isinstance(v, int | float | str | np.integer | np.floating)
        }
        manifest["images"].append(dict(stem=stem, band=band, metadata=meta_row, psf="psf.psf", targets=kept))
        n_star = sum(t["kind"] == "star" for t in kept)
        print(f"{stem}: {n_star} stars, {len(kept) - n_star} blanks")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    size = sum(p.stat().st_size for p in OUT_DIR.rglob("*") if p.is_file())
    print(f"wrote {OUT_DIR} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
