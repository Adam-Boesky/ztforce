"""Tests for ztforce.cache."""

from pathlib import Path


def test_make_cache_default_root(tmp_path, monkeypatch):
    """make_cache() defaults to ~/.ztforce/cache when root is None."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from ztforce.cache import make_cache

    cache = make_cache(None)
    assert cache.root == tmp_path / ".ztforce" / "cache"


def test_make_cache_explicit_root(tmp_path):
    """make_cache() accepts an explicit root path."""
    from ztforce.cache import make_cache

    cache = make_cache(tmp_path / "mydata")
    assert cache.root == tmp_path / "mydata"


def test_lightcurve_path_encoding(cache):
    """lightcurve_path encodes ra/dec into the directory and band into the filename."""
    from ztforce.cache import lightcurve_path

    p = lightcurve_path(cache, 150.12345, -2.54321, "r")
    assert p.suffix == ".ecsv"
    assert p.name == "r.ecsv"
    assert "150.12345" in str(p)
    assert "-2.54321" in str(p)


def test_different_bands_give_different_paths(cache):
    """Different bands produce different lightcurve paths."""
    from ztforce.cache import lightcurve_path

    pg = lightcurve_path(cache, 150.0, 2.0, "g")
    pr = lightcurve_path(cache, 150.0, 2.0, "r")
    assert pg != pr


def test_lightcurve_directory_created_on_first_access(cache):
    """lightcurve_path creates parent directories on first call."""
    from ztforce.cache import lightcurve_path

    lc = lightcurve_path(cache, 0.0, 0.0, "g")
    assert lc.parent.exists()
