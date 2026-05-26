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


def test_fits_path_encoding(cache):
    """fits_path encodes field/ccdid/qid/band/obsjd into the filename."""
    from ztforce.cache import fits_path

    p = fits_path(cache, 468, 3, 2, "g", 2459271.12345)
    assert p.suffix == ".fits"
    assert "000468" in p.name
    assert "_03_" in p.name
    assert "_2_" in p.name
    assert "_g_" in p.name
    assert "2459271" in p.name


def test_psf_path_encoding(cache):
    """psf_path produces .psf files in the psf/ subdirectory."""
    from ztforce.cache import psf_path

    p = psf_path(cache, 468, 3, 2, "g", 2459271.12345)
    assert p.suffix == ".psf"
    assert p.parent.name == "psf"
    assert "000468" in p.name


def test_lightcurve_path_encoding(cache):
    """lightcurve_path encodes ra/dec into the directory and band into the filename."""
    from ztforce.cache import lightcurve_path

    p = lightcurve_path(cache, 150.12345, -2.54321, "r")
    assert p.suffix == ".ecsv"
    assert p.name == "r.ecsv"
    assert "150.12345" in str(p)
    assert "-2.54321" in str(p)


def test_paths_are_deterministic(cache):
    """Same inputs always produce the same path."""
    from ztforce.cache import fits_path

    p1 = fits_path(cache, 100, 1, 1, "g", 2458000.0)
    p2 = fits_path(cache, 100, 1, 1, "g", 2458000.0)
    assert p1 == p2


def test_different_bands_give_different_paths(cache):
    """Different bands produce different lightcurve paths."""
    from ztforce.cache import lightcurve_path

    pg = lightcurve_path(cache, 150.0, 2.0, "g")
    pr = lightcurve_path(cache, 150.0, 2.0, "r")
    assert pg != pr


def test_directories_created_on_first_access(cache):
    """Parent directories are created when a path function is first called."""
    from ztforce.cache import fits_path, lightcurve_path, psf_path

    for fn in [fits_path, psf_path]:
        p = fn(cache, 1, 1, 1, "g", 1.0)
        assert p.parent.exists()

    lc = lightcurve_path(cache, 0.0, 0.0, "g")
    assert lc.parent.exists()
