"""Shared pytest fixtures for ztforce tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS


@pytest.fixture()
def mock_config(tmp_path, monkeypatch):
    """ZTForceConfig with mock credentials resolved from env vars."""
    monkeypatch.setenv("ZTFORCE_IRSA_USER", "testuser")
    monkeypatch.setenv("ZTFORCE_IRSA_PASS", "testpass")
    from ztforce.config import build_config

    return build_config(config_path=tmp_path / "nonexistent.toml")


@pytest.fixture()
def cache(tmp_path):
    """CacheConfig rooted at a temporary directory."""
    from ztforce.cache import make_cache

    return make_cache(tmp_path / "cache")


@pytest.fixture()
def synthetic_psf_file(tmp_path) -> Path:
    """Write a minimal valid DAOPhot PSF sidecar and return its path."""
    psf_size = 11
    n_tables = 3
    x_cen, y_cen = 1535.5, 1539.5
    norm_factor = 1000.0
    sigma = 1.5

    header = (
        f" GAUSSIAN  {psf_size:3d}    2    {n_tables}    0   14.000"
        f"  {norm_factor:12.3f}  {x_cen}  {y_cen}\n"
    )
    sigmas_line = f"  {sigma:.6E} {sigma:.6E}\n"

    c = psf_size // 2
    row_idx, col_idx = np.mgrid[0:psf_size, 0:psf_size]
    gauss = norm_factor * np.exp(-0.5 * ((col_idx - c) ** 2 / sigma**2 + (row_idx - c) ** 2 / sigma**2))
    # T1/T2 use gradient patterns so they don't cancel after normalization
    t1 = (col_idx - c) / (c + 1) * gauss * 0.2  # x-gradient × Gaussian
    t2 = (row_idx - c) / (c + 1) * gauss * 0.2  # y-gradient × Gaussian
    tables = [np.zeros((psf_size, psf_size)), t1, t2]

    def _fmt(t):
        flat = t.flatten()
        rows = []
        for i in range(0, len(flat), 6):
            rows.append("  " + " ".join(f"{v:.6E}" for v in flat[i : i + 6]))
        return "\n".join(rows) + "\n"

    path = tmp_path / "test.psf"
    with open(path, "w") as f:
        f.write(header)
        f.write(sigmas_line)
        for t in tables:
            f.write(_fmt(t))
    return path


@pytest.fixture()
def synthetic_fits_file(tmp_path) -> tuple[Path, int, int]:
    """Write a synthetic ZTF-like FITS image and return (path, cx, cy)."""
    size = 256
    cx, cy = size // 2, size // 2
    fwhm_px = 3.0
    sigma = fwhm_px / 2.355

    rng = np.random.default_rng(0)
    data = rng.normal(0, 10, (size, size)).astype(np.float32)
    y, x = np.mgrid[0:size, 0:size]
    data += (5000.0 * np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / sigma**2)).astype(np.float32)

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [cx + 1, cy + 1]
    wcs.wcs.cdelt = [-0.000281, 0.000281]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdr = wcs.to_header()
    hdr["MAGZP"] = 26.3
    hdr["OBSJD"] = 2459000.0
    hdr["GAIN"] = 6.2
    hdr["MEDFWHM"] = fwhm_px
    hdr["MAGLIM"] = 21.0
    hdr["RADESYS"] = "ICRS"

    path = tmp_path / "synth.fits"
    fits.writeto(str(path), data, hdr, overwrite=True)
    return path, cx, cy
