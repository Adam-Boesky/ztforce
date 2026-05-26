"""Network regression test: ztforce g-band lightcurve vs. PS1 DR2 photometry.

Run manually before releases:

    pytest -m network tests/ztforce/integration/test_regression.py -v

Excluded from CI by default (addopts: -m 'not network').

Star: RA=130.13113 Dec=+19.69525 (field near Praesepe/M44)
PS1 DR2 reference: g=15.824 ± 0.003 (65 detections, non-variable)
"""

from __future__ import annotations

import numpy as np
import pytest

# Reference photometry from PS1 DR2 mean catalog (queried 2026-05-26)
_RA = 130.13113
_DEC = 19.69525
_PS1_G = 15.824
_MAG_TOL = 0.10  # maximum allowed offset from PS1
_SCATTER_MAX = 0.05  # maximum allowed epoch-to-epoch scatter
_MIN_DETECTIONS = 3  # SNR > 3 epochs required


@pytest.mark.network
def test_gband_lightcurve_matches_ps1(tmp_path):
    """Median g-band magnitude from ztforce is within 0.10 mag of PS1 DR2."""
    from ztforce import build_config, run_forced_photometry

    config = build_config()
    lcs = run_forced_photometry(
        ra=_RA,
        dec=_DEC,
        bands=["g"],
        data_dir=str(tmp_path),
        config=config,
        n_download_workers=4,
        n_psf_workers=2,
    )
    assert "g" in lcs, "No g-band lightcurve returned"

    lc = lcs["g"]
    df = lc.df

    # Require at least _MIN_DETECTIONS epochs above SNR threshold
    detections = df[(df["flags"] == 0) & (~df["mag"].isna()) & (df["mag"] < 99)]
    assert (
        len(detections) >= _MIN_DETECTIONS
    ), f"Only {len(detections)} clean detections; expected ≥{_MIN_DETECTIONS}"

    median_g = float(np.median(detections["mag"].values))
    scatter = float(np.std(detections["mag"].values))

    assert abs(median_g - _PS1_G) < _MAG_TOL, (
        f"Median g={median_g:.3f} deviates from PS1 g={_PS1_G:.3f} by "
        f"{abs(median_g - _PS1_G):.3f} mag (tolerance={_MAG_TOL})"
    )
    assert scatter < _SCATTER_MAX, f"Epoch-to-epoch scatter {scatter:.3f} mag exceeds {_SCATTER_MAX} mag"
