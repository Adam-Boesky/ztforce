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


def lightcurve_path(cache: CacheConfig, ra: float, dec: float, band: str) -> Path:
    """Path for a cached per-source lightcurve ECSV."""
    coord_dir = f"{ra:.5f}_{dec:.5f}"
    p = cache.root / "lightcurves" / coord_dir / f"{band}.ecsv"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
