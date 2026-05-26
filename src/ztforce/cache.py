"""Cache directory layout and path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CacheConfig:
    """Root cache directory for all ztforce data."""

    root: Path


def make_cache(root: str | Path | None = None) -> CacheConfig:
    """Create a CacheConfig, defaulting to ~/.ztforce/cache."""
    if root is None:
        root = Path.home() / ".ztforce" / "cache"
    return CacheConfig(root=Path(root))


def _image_stem(field: int, ccdid: int, qid: int, band: str, obsjd: float) -> str:
    return f"{field:06d}_{ccdid:02d}_{qid}_{band}_{obsjd:.5f}"


def fits_path(cache: CacheConfig, field: int, ccdid: int, qid: int, band: str, obsjd: float) -> Path:
    """Path for a cached ZTF science FITS image."""
    p = cache.root / "fits" / (_image_stem(field, ccdid, qid, band, obsjd) + ".fits")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def psf_path(cache: CacheConfig, field: int, ccdid: int, qid: int, band: str, obsjd: float) -> Path:
    """Path for a cached DAOPhot PSF sidecar file."""
    p = cache.root / "psf" / (_image_stem(field, ccdid, qid, band, obsjd) + ".psf")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def lightcurve_path(cache: CacheConfig, ra: float, dec: float, band: str) -> Path:
    """Path for a cached per-source lightcurve ECSV."""
    coord_dir = f"{ra:.5f}_{dec:.5f}"
    p = cache.root / "lightcurves" / coord_dir / f"{band}.ecsv"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
