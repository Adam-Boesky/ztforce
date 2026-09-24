"""Accuracy of ztforce's forced photometry against the ZTF pipeline's own PSF catalog.

The fixture (``tests/data/psfcat_agreement``, built by ``scripts/build_psfcat_fixture.py``)
holds real IBE cutouts around isolated, unsaturated psfcat stars spread over each
quadrant, plus blank-sky positions, for several epochs, bands and field/CCD/quadrants.

psfcat fluxes come from the same pixels, the same DAOPhot PSF sidecar and the same
MAGZP, so zero points and colour terms cancel: a correct forced fit reproduces them to
~1%.  psfcat ``sigflux`` includes a per-image systematic floor, so uncertainties are
checked on blank sky (where the true flux is zero) rather than against the catalog.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ztforce.config import build_config

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "psfcat_agreement"

# Pass criteria
MAX_MEDIAN_FLUX_BIAS = 0.01  # |median(ztforce / psfcat) - 1|
MAX_FLUX_RATIO_SCATTER = 0.02  # robust sigma of ztforce / psfcat
MAX_IMAGE_MEDIAN_BIAS = 0.03  # worst single image: catches per-quadrant/epoch offsets
MAX_MEDIAN_CHISQ = 1.5  # isolated stars should be fit to within the noise
MAX_BLANK_MEDIAN_PULL = 0.3  # blank sky: flux / flux_err centred on zero...
BLANK_PULL_SIGMA_RANGE = (0.8, 1.3)  # ...with unit spread


def _robust_sigma(x) -> float:
    p16, p84 = np.percentile(x, [16, 84])
    return float(0.5 * (p84 - p16))


@functools.cache
def _results() -> pd.DataFrame:
    """Run ztforce's per-epoch fit on every fixture target (once per session)."""
    from ztforce.pipeline import _process_one_epoch

    # Credentials are never used here (all files are local); build_config just needs some.
    config = build_config(irsa_user="offline", irsa_pass="offline", config_path=FIXTURE / "none.toml")
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    rows = []
    for image in manifest["images"]:
        md = image["metadata"]
        image_id = f"{int(md['field'])}-{int(md['ccdid'])}-{int(md['qid'])}-{float(md['obsjd']):.3f}"
        img_dir = FIXTURE / image["stem"]
        for t in image["targets"]:
            res = _process_one_epoch(
                str(img_dir / t["file"]),
                str(img_dir / image["psf"]),
                t["ra"],
                t["dec"],
                image["band"],
                image_id,
                config,
                (md["crpix1"], md["crpix2"]),
            )
            rows.append(
                dict(
                    image=image["stem"],
                    kind=t["kind"],
                    xpos=t["xpos"],
                    ypos=t["ypos"],
                    cat_flux=t.get("flux", np.nan),
                    flux=res["flux"],
                    flux_err=res["flux_err"],
                    chisq=res.get("chisq", np.nan),
                    fit_failed=bool(res["flags"] & 3),
                )
            )
    return pd.DataFrame(rows)


def _stars() -> pd.DataFrame:
    df = _results()
    stars = df[df["kind"] == "star"].copy()
    stars["ratio"] = stars["flux"] / stars["cat_flux"]
    return stars


def test_fixture_present():
    """The fixture covers several images, bands and quadrants."""
    stars = _stars()
    assert len(stars) >= 60
    assert stars["image"].nunique() >= 8
    assert stars["image"].str[0].nunique() == 2  # g and r


def test_every_target_is_fitted():
    """No fixture target fails to fit (edge / NaN / processing error)."""
    failed = _results()[_results()["fit_failed"]]
    assert failed.empty, f"{len(failed)} targets failed to fit:\n{failed[['image', 'kind', 'xpos', 'ypos']]}"


def test_flux_matches_psfcat():
    """Forced-fit fluxes of isolated stars reproduce psfcat to ~1%, with ~2% scatter."""
    stars = _stars()
    ratio = stars["ratio"].dropna()
    median, scatter = float(np.median(ratio)), _robust_sigma(ratio)
    per_image = stars.groupby("image")["ratio"].median()
    summary = (
        f"median ratio {median:.4f}, robust scatter {scatter:.4f}, "
        f"per-image medians {per_image.min():.3f}-{per_image.max():.3f}"
    )
    assert abs(median - 1) < MAX_MEDIAN_FLUX_BIAS, summary
    assert scatter < MAX_FLUX_RATIO_SCATTER, summary
    assert (per_image - 1).abs().max() < MAX_IMAGE_MEDIAN_BIAS, (
        summary + "\n" + per_image.round(4).to_string()
    )


def test_flux_ratio_independent_of_quadrant_position():
    """The PSF varies over the quadrant; the flux ratio must not (catches wrong PSF coordinates)."""
    stars = _stars().dropna(subset=["ratio"])
    # Compare stars in the outer ring of the quadrant with those near its centre.
    r = np.hypot(stars["xpos"] - 1536, stars["ypos"] - 1540)
    inner, outer = stars[r < 900]["ratio"], stars[r > 1300]["ratio"]
    assert len(inner) >= 5 and len(outer) >= 5
    diff = float(np.median(outer) - np.median(inner))
    assert abs(diff) < MAX_MEDIAN_FLUX_BIAS, f"outer - inner median ratio {diff:+.4f}"


def test_isolated_star_chisq_near_one():
    """An isolated star fit with the right PSF has reduced chi-squared ~1."""
    chisq = _stars()["chisq"].dropna()
    assert len(chisq) > 0
    assert float(np.median(chisq)) < MAX_MEDIAN_CHISQ, f"median chisq {np.median(chisq):.2f}"


def test_blank_sky_is_consistent_with_zero():
    """Blank sky gives zero flux within errors, and the errors describe the scatter."""
    blanks = _results()[_results()["kind"] == "blank"]
    pull = (blanks["flux"] / blanks["flux_err"]).dropna()
    assert len(pull) >= 20
    median, sigma = float(np.median(pull)), _robust_sigma(pull)
    summary = f"blank-sky pull median {median:+.2f}, robust sigma {sigma:.2f} (n={len(pull)})"
    assert abs(median) < MAX_BLANK_MEDIAN_PULL, summary
    assert BLANK_PULL_SIGMA_RANGE[0] < sigma < BLANK_PULL_SIGMA_RANGE[1], summary


if __name__ == "__main__":
    # Print the scoreboard without pytest: python tests/ztforce/integration/test_psfcat_agreement.py
    stars, res = _stars(), _results()
    blanks = res[res["kind"] == "blank"]
    pull = blanks["flux"] / blanks["flux_err"]
    print(
        f"stars: n={len(stars)}  median ztforce/psfcat {stars['ratio'].median():.4f}  "
        f"robust scatter {_robust_sigma(stars['ratio'].dropna()):.4f}  "
        f"median chisq {stars['chisq'].median():.2f}"
    )
    print(
        f"blank: n={len(blanks)}  pull median {pull.median():+.2f}  "
        f"robust sigma {_robust_sigma(pull.dropna()):.2f}"
    )
    print("per image:")
    print(
        stars.groupby("image")
        .agg(n=("ratio", "size"), median_ratio=("ratio", "median"), median_chisq=("chisq", "median"))
        .round(3)
        .to_string()
    )
    pytest.main([__file__, "-q", "-p", "no:cacheprovider"])
