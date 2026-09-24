"""Tests for ztforce.lightcurve (Lightcurve class)."""

from __future__ import annotations

import numpy as np
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_lc(ra=150.0, dec=2.0):
    from ztforce.lightcurve import Lightcurve

    return Lightcurve(ra=ra, dec=dec)


# Test epochs default to zero point 26.3; stacked fluxes come back on STACK_ZERO_POINT.
_ZP = 26.3
_K = 10 ** (-0.4 * (_ZP - 25.0))


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
    """A good non-detection gets the SNU-sigma limit at the target; the image depth is kept too."""
    from ztforce.lightcurve import SNU

    lc = _make_lc()
    _add_non_detection(lc, flux=5.0, flux_err=50.0, zp=26.3)
    row = lc.df.iloc[0]
    assert row["upper_limit"] == pytest.approx(26.3 - 2.5 * np.log10(SNU * 50.0))
    assert row["mag_limit"] == pytest.approx(21.0)


def test_upper_limit_nan_for_flagged_epoch():
    """A flagged epoch is not a usable non-detection, so it gets no upper limit."""
    from ztforce.utils import flux_to_ab_mag

    lc = _make_lc()
    mag, merr = flux_to_ab_mag(5.0, 26.3, 50.0)
    lc.add_epoch(2459000.0, "g", 5.0, 50.0, mag, merr, 26.3, flags=4, mag_limit=21.0)
    assert np.isnan(lc.df.iloc[0]["upper_limit"])


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
    assert result.loc["g", "flux_stack"] == pytest.approx(1000.0 * _K)
    assert result.loc["g", "flux_err_stack"] == pytest.approx(100.0 * _K)


def test_stack_ivw_two_detections():
    """IVW of two equal-error epochs equals their arithmetic mean."""
    lc = _make_lc()
    _add_detection(lc, obsjd=2459000.0, flux=1000.0, flux_err=100.0)
    _add_detection(lc, obsjd=2459001.0, flux=2000.0, flux_err=100.0)
    result = lc.stack()
    # IVW with equal errors → arithmetic mean
    assert result.loc["g", "flux_stack"] == pytest.approx(1500.0 * _K, rel=1e-6)
    # Error = 1/sqrt(2) * 100
    assert result.loc["g", "flux_err_stack"] == pytest.approx((100.0 / np.sqrt(2)) * _K, rel=1e-6)


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
    assert result.loc["g", "flux_stack"] == pytest.approx(expected_flux * _K, rel=1e-6)
    assert result.loc["g", "flux_err_stack"] == pytest.approx(expected_err * _K, rel=1e-6)


def test_stack_includes_non_detections():
    """stack() uses every good epoch, detected or not (ZFPS guide section 6.6)."""
    lc = _make_lc()
    _add_detection(lc, obsjd=2459000.0, flux=1000.0, flux_err=100.0)
    _add_non_detection(lc, obsjd=2459001.0, flux=1.0, flux_err=100.0)
    result = lc.stack()
    assert result.loc["g", "n_epochs"] == 2
    assert result.loc["g", "flux_stack"] == pytest.approx(500.5 * _K)


def test_stack_excludes_flagged_epochs():
    """Epochs with nonzero quality flags never enter a stack."""
    from ztforce.utils import flux_to_ab_mag

    lc = _make_lc()
    _add_detection(lc, obsjd=2459000.0, flux=1000.0, flux_err=100.0)
    mag, merr = flux_to_ab_mag(9000.0, _ZP, 100.0)
    lc.add_epoch(2459001.0, "g", 9000.0, 100.0, mag, merr, _ZP, flags=4)  # bad calibration
    result = lc.stack()
    assert result.loc["g", "n_epochs"] == 1
    assert result.loc["g", "flux_stack"] == pytest.approx(1000.0 * _K)


def test_stack_low_snr_gives_upper_limit():
    """A stack below SNT has no magnitude and an SNU-sigma upper limit instead."""
    from ztforce.lightcurve import SNU, STACK_ZERO_POINT

    lc = _make_lc()
    _add_non_detection(lc, obsjd=2459000.0, flux=10.0, flux_err=100.0)
    row = lc.stack().loc["g"]
    assert not row["detection"]
    assert row["snr_stack"] == pytest.approx(0.1)
    assert np.isnan(row["mag_stack"]) and np.isnan(row["mag_err_stack"])
    assert row["upper_limit_stack"] == pytest.approx(STACK_ZERO_POINT - 2.5 * np.log10(SNU * 100.0 * _K))


def test_stack_of_faint_epochs_can_be_a_detection():
    """Individually undetected epochs can add up to a detected stack."""
    lc = _make_lc()
    for i in range(25):  # S/N 1 each -> S/N 5 stacked
        _add_non_detection(lc, obsjd=2459000.0 + i, flux=100.0, flux_err=100.0)
    row = lc.stack().loc["g"]
    assert not lc.df["detection"].any()
    assert row["detection"]
    assert row["snr_stack"] == pytest.approx(5.0)
    assert np.isnan(row["upper_limit_stack"])


def test_stack_jd_window():
    """stack() respects jd_min and jd_max boundaries."""
    lc = _make_lc()
    _add_detection(lc, obsjd=2459000.0, flux=500.0, flux_err=50.0)
    _add_detection(lc, obsjd=2459100.0, flux=1000.0, flux_err=50.0)
    result = lc.stack(jd_min=2459050.0)
    assert result.loc["g", "n_epochs"] == 1
    assert result.loc["g", "flux_stack"] == pytest.approx(1000.0 * _K)


def test_stack_empty_band_omitted():
    """Bands with no good epochs are not present in stack result."""
    from ztforce.utils import flux_to_ab_mag

    lc = _make_lc()
    _add_detection(lc, band="g")
    mag, merr = flux_to_ab_mag(1000.0, _ZP, 50.0)
    lc.add_epoch(2459001.0, "r", 1000.0, 50.0, mag, merr, _ZP, flags=16)  # bad seeing
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


@pytest.mark.parametrize("span", [800.0, 1000.0, 1095.0, 1100.0])
def test_rolling_stack_days_covers_every_epoch_once(span):
    """Non-overlapping time windows stack every epoch exactly once, including the newest."""
    lc = _make_lc()
    jds = np.linspace(2459000.0, 2459000.0 + span, 60)
    for jd in jds:
        _add_detection(lc, obsjd=float(jd))
    result = lc.rolling_stack(window=365.0, window_unit="days", step=365.0)
    assert result["n_epochs"].sum() == len(jds)


def test_rolling_stack_images_unit_returns_expected_columns():
    """rolling_stack with window_unit='images' returns expected columns."""
    lc = _make_lc()
    for i in range(20):
        _add_detection(lc, obsjd=2459000.0 + i * 3, flux=1000.0, flux_err=100.0)
    result = lc.rolling_stack(window=6, window_unit="images")
    assert not result.empty
    for col in (
        "obsjd_center",
        "band",
        "flux_stack",
        "flux_err_stack",
        "mag_stack",
        "mag_err_stack",
        "n_epochs",
    ):
        assert col in result.columns


def test_rolling_stack_images_ivw_analytic():
    """Image-windowed rolling_stack IVW matches hand-computed Σ(f/σ²)/Σ(1/σ²)."""
    lc = _make_lc()
    fluxes = [800.0, 1200.0, 1000.0, 900.0, 1100.0]
    errors = [50.0, 100.0, 80.0, 60.0, 90.0]
    for i, (f, e) in enumerate(zip(fluxes, errors, strict=True)):
        _add_detection(lc, obsjd=2459000.0 + i, flux=f, flux_err=e)

    # window=3, step=1 → centre index 1 covers indices 0,1,2
    result = lc.rolling_stack(window=3, window_unit="images", step=1)
    first = result[result["band"] == "g"].sort_values("obsjd_center").iloc[0]

    inv_var = [1 / e**2 for e in errors[:3]]
    expected_flux = sum(f * iv for f, iv in zip(fluxes[:3], inv_var, strict=True)) / sum(inv_var)
    expected_err = 1.0 / np.sqrt(sum(inv_var))

    assert first["flux_stack"] == pytest.approx(expected_flux * _K, rel=1e-6)
    assert first["flux_err_stack"] == pytest.approx(expected_err * _K, rel=1e-6)


@pytest.mark.parametrize(("n", "window", "step"), [(10, 4, None), (10, 5, None), (11, 4, 3), (7, 7, None)])
def test_rolling_stack_images_exact_windows_cover_every_epoch(n, window, step):
    """Every image window holds exactly `window` epochs, and together they cover all of them."""
    lc = _make_lc()
    for i in range(n):
        _add_detection(lc, obsjd=2459000.0 + i)
    result = lc.rolling_stack(window=window, window_unit="images", step=step)
    assert (result["n_epochs"] == window).all()
    # The newest epoch is in the last window: its IVW centre sits within the last `window` epochs.
    assert result["obsjd_center"].max() > 2459000.0 + n - window


def test_rolling_stack_images_skips_flagged_epochs():
    """Flagged epochs neither enter nor use up an image window."""
    from ztforce.utils import flux_to_ab_mag

    lc = _make_lc()
    for i in range(6):
        _add_detection(lc, obsjd=2459000.0 + i)
        mag, merr = flux_to_ab_mag(1000.0, _ZP, 50.0)
        lc.add_epoch(2459000.5 + i, "g", 1000.0, 50.0, mag, merr, _ZP, flags=4)
    result = lc.rolling_stack(window=3, window_unit="images", step=3)
    assert list(result["n_epochs"]) == [3, 3]


def test_rolling_stack_images_short_band_gives_one_window():
    """A band with fewer good epochs than the window still gets a stack of all of them."""
    lc = _make_lc()
    for i in range(2):
        _add_detection(lc, obsjd=2459000.0 + i)
    result = lc.rolling_stack(window=5, window_unit="images")
    assert list(result["n_epochs"]) == [2]


def test_rolling_stack_obsjd_center_is_ivw_weighted_mean():
    """obsjd_center is the IVW-weighted mean JD of the window, not the arithmetic midpoint."""
    lc = _make_lc()
    # Three epochs; first has 100× lower error than the others → dominates the centre.
    _add_detection(lc, obsjd=2459000.0, flux=1000.0, flux_err=10.0)  # weight = 1/100
    _add_detection(lc, obsjd=2459010.0, flux=1000.0, flux_err=1000.0)  # weight = 1/1e6
    _add_detection(lc, obsjd=2459020.0, flux=1000.0, flux_err=1000.0)  # weight = 1/1e6

    # window=3, half=1 → range(1, 2, 1): one window covering all three epochs
    result = lc.rolling_stack(window=3, window_unit="images", step=1)
    row = result[result["band"] == "g"].iloc[0]

    jds = [2459000.0, 2459010.0, 2459020.0]
    errs = [10.0, 1000.0, 1000.0]
    inv_var = [1 / e**2 for e in errs]
    expected_center = sum(jd * iv for jd, iv in zip(jds, inv_var, strict=True)) / sum(inv_var)

    assert row["obsjd_center"] == pytest.approx(expected_center, rel=1e-6)
    # IVW centre should be near 2459000, not the arithmetic midpoint 2459010.
    assert abs(row["obsjd_center"] - 2459000.0) < 1.0


def test_stack_has_no_obsjd_center_column():
    """stack() output does not include obsjd_center (it belongs only to rolling_stack)."""
    lc = _make_lc()
    _add_detection(lc)
    result = lc.stack()
    assert "obsjd_center" not in result.columns


def test_stack_mixed_zero_points_recovers_true_mag():
    """Epochs of one constant source taken at different zero points stack to that source's magnitude."""
    from ztforce.lightcurve import STACK_ZERO_POINT

    lc = _make_lc()
    true_mag = 18.0
    # Same source, different transparency: the instrumental flux scales with the zero point.
    for i, zp in enumerate([26.3, 25.4, 26.1, 25.7]):
        flux = 10 ** (-0.4 * (true_mag - zp))
        _add_detection(lc, obsjd=2459000.0 + i, flux=flux, flux_err=flux / 20, zp=zp)

    result = lc.stack()
    assert result.loc["g", "mag_stack"] == pytest.approx(true_mag, abs=1e-9)
    assert result.loc["g", "flux_stack"] == pytest.approx(10 ** (-0.4 * (true_mag - STACK_ZERO_POINT)))


def test_rolling_stack_mixed_zero_points_is_flat_for_constant_source():
    """A constant source observed through changing zero points yields a flat rolling stack."""
    lc = _make_lc()
    true_mag = 19.0
    zps = [26.3, 25.4, 26.1, 25.7, 26.2, 25.5, 26.0, 25.8]
    for i, zp in enumerate(zps):
        flux = 10 ** (-0.4 * (true_mag - zp))
        _add_detection(lc, obsjd=2459000.0 + 10 * i, flux=flux, flux_err=flux / 20, zp=zp)

    result = lc.rolling_stack(window=3, window_unit="images", step=1)
    assert np.allclose(result["mag_stack"], true_mag, atol=1e-9)


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
