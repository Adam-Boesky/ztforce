"""Lightcurve: per-epoch storage, stacking, and I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

from .utils import flux_to_ab_mag

_BAND_ORDER = ["g", "r", "i"]
SNT = 3.0  # detection signal-to-noise threshold
SNU = 5.0  # signal-to-noise of the upper limit quoted for a non-detection (ZFPS guide section 6.4)
# Stacked fluxes are expressed on this AB zero point.  Each epoch's instrumental flux
# is rescaled to it before stacking, since ZTF zero points vary by up to ~1 mag
# between exposures and averaging raw counts across them biases the stack.
STACK_ZERO_POINT = 25.0


def _nan_if_none(v: float | None) -> float:
    return float(v) if v is not None else float("nan")


class Lightcurve:
    """Per-source forced-photometry lightcurve in absolute AB magnitudes.

    Uncertainties are statistical only (sky + Poisson), so they are underestimated:
    they omit calibration/PSF systematics (bright sources) and, since the fit is on
    science images rather than difference images, residual host light that varies
    with seeing (extended hosts).  ``chisq`` is ~1 for an isolated point source and
    flags extended, blended, or poorly fit sources; it is not used to rescale errors.
    """

    def __init__(self, ra: float, dec: float) -> None:
        self.ra = ra
        self.dec = dec
        self._rows: list[dict] = []
        self.cache_key: str = ""

    # ── I/O ──────────────────────────────────────────────────────────────────

    def add_epoch(
        self,
        obsjd: float,
        band: str,
        flux: float,
        flux_err: float,
        mag: float,
        mag_err: float,
        zero_point: float,
        flags: int,
        x_fit: float | None = None,
        y_fit: float | None = None,
        mag_limit: float | None = None,
        image_id: str | None = None,
        chisq: float | None = None,
        infobits: int | None = None,
        seeing: float | None = None,
        scisigpix: float | None = None,
    ) -> None:
        """Append one exposure's measurement.

        ``flags`` is the quality bitmask (see ``ztforce._constants``); only epochs with
        ``flags == 0`` can be detections or enter a stack.
        """
        snr = flux / flux_err if flux_err and flux_err > 0 else float("nan")
        is_det = np.isfinite(snr) and snr >= SNT and flags == 0
        # A good non-detection gets an SNU-sigma limit at the target position (ZFPS guide
        # section 6.4); mag_limit is the image-wide 5-sigma depth, kept alongside.
        if not is_det and flags == 0 and flux_err and flux_err > 0:
            upper_limit = float(zero_point - 2.5 * np.log10(SNU * flux_err))
        else:
            upper_limit = float("nan")

        self._rows.append(
            dict(
                obsjd=obsjd,
                band=band,
                flux=flux,
                flux_err=flux_err,
                mag=mag,
                mag_err=mag_err,
                zero_point=zero_point,
                flags=flags,
                snr=snr,
                detection=is_det,
                upper_limit=upper_limit,
                mag_limit=_nan_if_none(mag_limit),
                x_fit=x_fit if x_fit is not None else float("nan"),
                y_fit=y_fit if y_fit is not None else float("nan"),
                image_id=image_id or "",
                chisq=_nan_if_none(chisq),
                infobits=infobits if infobits is not None else -1,
                seeing=_nan_if_none(seeing),
                scisigpix=_nan_if_none(scisigpix),
            )
        )

    @property
    def df(self) -> pd.DataFrame:
        """All epochs as a DataFrame, sorted by obsjd."""
        return pd.DataFrame(self._rows).sort_values("obsjd").reset_index(drop=True)

    @property
    def bands(self) -> list[str]:
        """Unique bands present, in canonical g/r/i order."""
        present = {r["band"] for r in self._rows}
        return [b for b in _BAND_ORDER if b in present]

    def get_band(self, band: str) -> pd.DataFrame:
        """Return epochs for a single band, sorted by obsjd."""
        df = self.df
        return df[df["band"] == band].reset_index(drop=True)

    # ── Stacking ─────────────────────────────────────────────────────────────

    def stack(
        self,
        jd_min: float | None = None,
        jd_max: float | None = None,
        bands: list[str] | None = None,
    ) -> pd.DataFrame:
        """Inverse-variance-weighted stack of all good epochs within a JD window.

        Follows the ZFPS user guide (section 6.6): every unflagged epoch is stacked,
        detected or not, after rescaling to a common zero point.

        Returns a DataFrame indexed by band with columns:
          flux_stack, flux_err_stack, snr_stack, detection, mag_stack, mag_err_stack,
          upper_limit_stack, n_epochs.
        ``flux_stack`` and ``flux_err_stack`` are on the AB zero point ``STACK_ZERO_POINT``.
        A stack with ``snr_stack >= SNT`` is a detection with a magnitude; otherwise
        ``mag_stack`` is NaN and ``upper_limit_stack`` is its SNU-sigma limiting magnitude.
        """
        df = self.df
        if jd_min is not None:
            df = df[df["obsjd"] >= jd_min]
        if jd_max is not None:
            df = df[df["obsjd"] <= jd_max]
        target_bands = bands or self.bands

        records = []
        for band in target_bands:
            rec = self._stack_window(band, df[df["band"] == band])
            if rec is not None:
                records.append(rec)
        return pd.DataFrame(records).set_index("band").drop(columns="obsjd_center")

    def _stack_window(self, band: str, sub: pd.DataFrame) -> dict | None:
        """IVW stack of the good epochs in one window; None when there are none."""
        valid = sub[(sub["flags"] == 0) & np.isfinite(sub["flux"]) & (sub["flux_err"] > 0)]
        if valid.empty:
            return None
        # Put every epoch on a common zero point before averaging.
        scale = 10.0 ** (-0.4 * (valid["zero_point"] - STACK_ZERO_POINT))
        flux = valid["flux"] * scale
        inv_var = 1.0 / (valid["flux_err"] * scale) ** 2
        f_stack = float((flux * inv_var).sum() / inv_var.sum())
        e_stack = float(1.0 / np.sqrt(inv_var.sum()))
        jd_c = float((valid["obsjd"] * inv_var).sum() / inv_var.sum())
        snr = f_stack / e_stack
        is_det = snr >= SNT
        if is_det:
            mag, merr = flux_to_ab_mag(f_stack, STACK_ZERO_POINT, e_stack)
            upper_limit = float("nan")
        else:
            mag = merr = float("nan")
            upper_limit = STACK_ZERO_POINT - 2.5 * np.log10(SNU * e_stack)
        return dict(
            obsjd_center=jd_c,
            band=band,
            flux_stack=f_stack,
            flux_err_stack=e_stack,
            snr_stack=snr,
            detection=is_det,
            mag_stack=float(mag),
            mag_err_stack=float(merr) if merr is not None else float("nan"),
            upper_limit_stack=float(upper_limit),
            n_epochs=len(valid),
        )

    def rolling_stack(
        self,
        window: float,
        window_unit: str = "days",
        bands: list[str] | None = None,
        step: float | None = None,
    ) -> pd.DataFrame:
        """Rolling IVW stack in a sliding window.

        Args:
            window: Width of the rolling window in the units given by ``window_unit``.
            window_unit: ``'days'`` or ``'years'`` for time-based windows;
                ``'images'`` for a fixed epoch count regardless of cadence.
            bands: Bands to include (default: all present).
            step: Step between window centres in the same unit as ``window``.
                Defaults to ``window / 2`` (50 % overlap).

        Returns:
            Long-format DataFrame with the columns of :meth:`stack` plus ``obsjd_center``,
            the inverse-variance-weighted mean JD of the window's epochs.
        """
        target_bands = bands or self.bands
        if window_unit == "days":
            win_days = window
        elif window_unit == "years":
            win_days = window * 365.25
        elif window_unit == "images":
            return self._rolling_stack_images(
                int(window), target_bands, int(step) if step is not None else None
            )
        else:
            raise ValueError(f"Unknown window_unit '{window_unit}'. Use 'days', 'years', or 'images'.")
        return self._rolling_stack_time(win_days, target_bands, step)

    def _rolling_stack_time(self, window_days: float, bands: list[str], step: float | None) -> pd.DataFrame:
        step = step or (window_days / 2)
        df = self.df
        # Windows are half-open, [c - half, c + half), so an epoch on a shared edge counts
        # once; add windows until the last one extends past the newest epoch.
        half = window_days / 2
        jd_min, jd_max = df["obsjd"].min(), df["obsjd"].max()
        n_extra = max(int(np.floor((jd_max - jd_min - window_days) / step)) + 1, 0)
        centers = jd_min + half + step * np.arange(n_extra + 1)

        records = []
        for jd_c in centers:
            sub = df[(df["obsjd"] >= jd_c - half) & (df["obsjd"] < jd_c + half)]
            for band in bands:
                rec = self._stack_window(band, sub[sub["band"] == band])
                if rec is not None:
                    records.append(rec)
        return pd.DataFrame(records)

    def _rolling_stack_images(self, window: int, bands: list[str], step: int | None) -> pd.DataFrame:
        step = step or max(1, window // 2)
        half = window // 2
        df = self.df

        records = []
        for band in bands:
            lc = df[df["band"] == band].sort_values("obsjd").reset_index(drop=True)
            for i in range(half, len(lc) - half, step):
                rec = self._stack_window(band, lc.iloc[i - half : i + half + 1])
                if rec is not None:
                    records.append(rec)
        return pd.DataFrame(records)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save to an Astropy ECSV file preserving all columns and metadata."""
        t = Table.from_pandas(self.df)
        t.meta["ra"] = self.ra
        t.meta["dec"] = self.dec
        t.meta["cache_key"] = self.cache_key
        t.write(str(path), format="ascii.ecsv", overwrite=True)

    @classmethod
    def load(cls, path: str | Path) -> Lightcurve:
        """Load from an Astropy ECSV file saved by save()."""
        t = Table.read(str(path), format="ascii.ecsv")
        lc = cls(ra=float(t.meta["ra"]), dec=float(t.meta["dec"]))
        lc.cache_key = t.meta.get("cache_key", "")
        lc._rows = t.to_pandas().to_dict("records")
        return lc

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of epochs."""
        return len(self._rows)

    def __repr__(self) -> str:
        """Short representation."""
        return (
            f"Lightcurve(ra={self.ra:.5f}, dec={self.dec:.5f}, " f"n_epochs={len(self)}, bands={self.bands})"
        )
