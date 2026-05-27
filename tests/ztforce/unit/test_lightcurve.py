"""Tests for ztforce.lightcurve (Lightcurve class)."""

from __future__ import annotations

import numpy as np
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_lc(ra=150.0, dec=2.0):
    from ztforce.lightcurve import Lightcurve

    return Lightcurve(ra=ra, dec=dec)


def _add_detection(lc, obsjd=2459000.0, band="g", flux=1000.0, flux_err=50.0, zp=26.3):
    """Add a clearly detected epoch (SNR = 20 by default)."""
    from ztforce.utils import flux_to_ab_mag

    mag, merr = flux_to_ab_mag(flux, zp, flux_err)
    lc.add_epoch(obsjd, band, flux, flux_err, mag, merr, zp, flags=0, mag_limit=21.0)


def _add_non_detection(lc, obsjd=2459100.0, band="g", flux=5.0, flux_err=50.0, zp=26.3):
    """Add a non-detection epoch (SNR = 0.1 by default)."""
    from ztforce.utils import flux_to_ab_mag

    mag, merr = flux_to_ab_mag(flux, zp, flux_err)
    lc.add_epoch(obsjd, band, flux, flux_err, mag, merr, zp, flags=0, mag_limit=21.0)


# ── Construction and add_epoch ────────────────────────────────────────────────


def test_empty_lightcurve_length():
    """Empty Lightcurve has length 0."""
    lc = _make_lc()
    assert len(lc) == 0


def test_add_epoch_increments_length():
    """add_epoch increases len by 1."""
    lc = _make_lc()
    _add_detection(lc)
    assert len(lc) == 1


def test_detection_flag_above_snt():
    """An epoch with SNR >= SNT and flags==0 is marked as a detection."""
    from ztforce.lightcurve import SNT

    lc = _make_lc()
    flux, ferr = 1000.0, 10.0  # SNR = 100 >> SNT=3
    assert flux / ferr >= SNT
    _add_detection(lc, flux=flux, flux_err=ferr)
    assert lc.df["detection"].iloc[0]


def test_non_detection_flag_below_snt():
    """An epoch with SNR < SNT is NOT marked as a detection."""
    from ztforce.lightcurve import SNT

    lc = _make_lc()
    flux, ferr = 1.0, 50.0  # SNR = 0.02 << SNT
    assert flux / ferr < SNT
    _add_non_detection(lc, flux=flux, flux_err=ferr)
    assert not lc.df["detection"].iloc[0]


def test_flagged_epoch_not_detection():
    """An epoch with flags != 0 is not a detection regardless of SNR."""
    from ztforce.lightcurve import Lightcurve
    from ztforce.utils import flux_to_ab_mag

    lc = Lightcurve(ra=0.0, dec=0.0)
    flux, ferr = 1000.0, 10.0
    mag, merr = flux_to_ab_mag(flux, 26.3, ferr)
    lc.add_epoch(2459000.0, "g", flux, ferr, mag, merr, 26.3, flags=1, mag_limit=21.0)
    assert not lc.df["detection"].iloc[0]


def test_upper_limit_set_for_non_detection():
    """Non-detection rows have upper_limit equal to the supplied mag_limit."""
    lc = _make_lc()
    _add_non_detection(lc, flux=1.0, flux_err=100.0)
    row = lc.df.iloc[0]
    assert not row["detection"]
    assert row["upper_limit"] == pytest.approx(21.0)


def test_upper_limit_nan_for_detection():
    """Detection rows have NaN upper_limit."""
    lc = _make_lc()
    _add_detection(lc)
    row = lc.df.iloc[0]
    assert row["detection"]
    assert np.isnan(row["upper_limit"])


# ── df / bands / get_band ─────────────────────────────────────────────────────


def test_df_sorted_by_obsjd():
    """df returns rows sorted by obsjd ascending."""
    lc = _make_lc()
    _add_detection(lc, obsjd=2459100.0)
    _add_detection(lc, obsjd=2459000.0)
    jds = lc.df["obsjd"].tolist()
    assert jds == sorted(jds)


def test_bands_canonical_order():
    """bands returns bands in g/r/i order regardless of insertion order."""
    lc = _make_lc()
    _add_detection(lc, band="i")
    _add_detection(lc, band="g")
    _add_detection(lc, band="r")
    assert lc.bands == ["g", "r", "i"]


def test_bands_only_present():
    """bands only returns bands that have epochs."""
    lc = _make_lc()
    _add_detection(lc, band="g")
    _add_detection(lc, band="r")
    assert "i" not in lc.bands
    assert lc.bands == ["g", "r"]


def test_get_band_filters_correctly():
    """get_band returns only the requested band's rows."""
    lc = _make_lc()
    _add_detection(lc, band="g", obsjd=2459001.0)
    _add_detection(lc, band="r", obsjd=2459002.0)
    g = lc.get_band("g")
    assert (g["band"] == "g").all()
    assert len(g) == 1


# ── stack ─────────────────────────────────────────────────────────────────────


def test_stack_single_detection():
    """Stacking a single detection returns the original flux and error."""
    lc = _make_lc()
    _add_detection(lc, flux=1000.0, flux_err=100.0)
    result = lc.stack()
    assert "g" in result.index
    assert result.loc["g", "flux_stack"] == pytest.approx(1000.0)
    assert result.loc["g", "flux_err_stack"] == pytest.approx(100.0)


def test_stack_ivw_two_detections():
    """IVW of two equal-error epochs equals their arithmetic mean."""
    lc = _make_lc()
    _add_detection(lc, obsjd=2459000.0, flux=1000.0, flux_err=100.0)
    _add_detection(lc, obsjd=2459001.0, flux=2000.0, flux_err=100.0)
    result = lc.stack()
    # IVW with equal errors → arithmetic mean
    assert result.loc["g", "flux_stack"] == pytest.approx(1500.0, rel=1e-6)
    # Error = 1/sqrt(2) * 100
    assert result.loc["g", "flux_err_stack"] == pytest.approx(100.0 / np.sqrt(2), rel=1e-6)


def test_stack_ivw_analytic():
    """IVW stack matches hand-computed Σ(f/σ²)/Σ(1/σ²)."""
    lc = _make_lc()
    fluxes = [800.0, 1200.0, 1000.0]
    errors = [50.0, 100.0, 80.0]
    for i, (f, e) in enumerate(zip(fluxes, errors, strict=False)):
        _add_detection(lc, obsjd=2459000.0 + i, flux=f, flux_err=e)

    inv_var = [1 / e**2 for e in errors]
    expected_flux = sum(f * iv for f, iv in zip(fluxes, inv_var, strict=False)) / sum(inv_var)
    expected_err = 1.0 / np.sqrt(sum(inv_var))

    result = lc.stack()
    assert result.loc["g", "flux_stack"] == pytest.approx(expected_flux, rel=1e-6)
    assert result.loc["g", "flux_err_stack"] == pytest.approx(expected_err, rel=1e-6)


def test_stack_ignores_non_detections():
    """stack() only uses detection rows."""
    lc = _make_lc()
    _add_detection(lc, obsjd=2459000.0, flux=1000.0, flux_err=100.0)
    _add_non_detection(lc, obsjd=2459001.0, flux=1.0, flux_err=100.0)
    result = lc.stack()
    assert result.loc["g", "n_epochs"] == 1
    assert result.loc["g", "flux_stack"] == pytest.approx(1000.0)


def test_stack_jd_window():
    """stack() respects jd_min and jd_max boundaries."""
    lc = _make_lc()
    _add_detection(lc, obsjd=2459000.0, flux=500.0, flux_err=50.0)
    _add_detection(lc, obsjd=2459100.0, flux=1000.0, flux_err=50.0)
    result = lc.stack(jd_min=2459050.0)
    assert result.loc["g", "n_epochs"] == 1
    assert result.loc["g", "flux_stack"] == pytest.approx(1000.0)


def test_stack_empty_band_omitted():
    """Bands with no detections are not present in stack result."""
    lc = _make_lc()
    _add_detection(lc, band="g")
    _add_non_detection(lc, band="r")
    result = lc.stack()
    assert "g" in result.index
    assert "r" not in result.index


# ── rolling_stack ─────────────────────────────────────────────────────────────


def test_rolling_stack_returns_dataframe():
    """rolling_stack returns a DataFrame with expected columns."""
    lc = _make_lc()
    for i in range(5):
        _add_detection(lc, obsjd=2459000.0 + i * 10, flux=1000.0, flux_err=100.0)
    result = lc.rolling_stack(window=15.0)
    assert "obsjd_center" in result.columns
    assert "flux_stack" in result.columns
    assert "band" in result.columns


def test_rolling_stack_years_unit():
    """rolling_stack accepts window_unit='years'."""
    lc = _make_lc()
    for i in range(4):
        _add_detection(lc, obsjd=2459000.0 + i * 100, flux=1000.0, flux_err=100.0)
    result = lc.rolling_stack(window=1.0, window_unit="years")
    assert not result.empty


def test_rolling_stack_bad_unit_raises():
    """rolling_stack raises ValueError for unknown window_unit."""
    lc = _make_lc()
    _add_detection(lc)
    with pytest.raises(ValueError, match="window_unit"):
        lc.rolling_stack(window=10.0, window_unit="weeks")


# ── save / load round-trip ────────────────────────────────────────────────────


def test_save_load_roundtrip(tmp_path):
    """Lightcurve saved to ECSV and loaded back matches the original."""
    lc = _make_lc(ra=150.12345, dec=-2.54321)
    _add_detection(lc, obsjd=2459000.0, flux=1000.0, flux_err=50.0)
    _add_non_detection(lc, obsjd=2459100.0)

    path = tmp_path / "lc.ecsv"
    lc.save(path)

    lc2 = _make_lc.__class__  # avoid re-import noise
    from ztforce.lightcurve import Lightcurve

    lc2 = Lightcurve.load(path)

    assert lc2.ra == pytest.approx(lc.ra)
    assert lc2.dec == pytest.approx(lc.dec)
    assert len(lc2) == len(lc)
    assert list(lc2.df["obsjd"]) == pytest.approx(list(lc.df["obsjd"]))
    assert list(lc2.df["flux"]) == pytest.approx(list(lc.df["flux"]))


def test_save_creates_valid_ecsv(tmp_path):
    """Saved file has .ecsv suffix and is non-empty."""
    lc = _make_lc()
    _add_detection(lc)
    path = tmp_path / "test.ecsv"
    lc.save(path)
    assert path.exists()
    assert path.stat().st_size > 0


# ── repr / len ────────────────────────────────────────────────────────────────


def test_repr_contains_ra_dec():
    """repr contains ra, dec, and epoch count."""
    lc = _make_lc(ra=123.456, dec=-7.89)
    _add_detection(lc)
    r = repr(lc)
    assert "123.456" in r
    assert "-7.89" in r
    assert "1" in r


def test_len_matches_epoch_count():
    """len() returns the number of epochs added."""
    lc = _make_lc()
    for i in range(5):
        _add_detection(lc, obsjd=2459000.0 + i)
    assert len(lc) == 5


# ── plot (smoke test) ─────────────────────────────────────────────────────────
