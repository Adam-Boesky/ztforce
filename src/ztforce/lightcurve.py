"""Lightcurve: per-epoch storage, stacking, and I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

from .utils import flux_to_ab_mag

_BAND_ORDER = ["g", "r", "i"]
SNT = 3.0  # detection signal-to-noise threshold


class Lightcurve:
    """Per-source forced-photometry lightcurve in absolute AB magnitudes."""

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
    ) -> None:
        """Append one exposure's measurement."""
        snr = flux / flux_err if flux_err and flux_err > 0 else float("nan")
        is_det = np.isfinite(snr) and snr >= SNT and flags == 0
        upper_limit = mag_limit if not is_det and mag_limit is not None else float("nan")

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
                x_fit=x_fit if x_fit is not None else float("nan"),
                y_fit=y_fit if y_fit is not None else float("nan"),
                image_id=image_id or "",
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
        """Inverse-variance-weighted stack of detections within a JD window.

        Returns a DataFrame indexed by band with columns:
          flux_stack, flux_err_stack, mag_stack, mag_err_stack, n_epochs.
        """
        df = self.df
        if jd_min is not None:
            df = df[df["obsjd"] >= jd_min]
        if jd_max is not None:
            df = df[df["obsjd"] <= jd_max]
        target_bands = bands or self.bands

        records = []
        for band in target_bands:
            sub = df[(df["band"] == band) & df["detection"]]
            if sub.empty:
                continue
            rec = self._stack_window(band, sub)
            if rec is not None:
                records.append(rec)
        return pd.DataFrame(records).set_index("band").drop(columns="obsjd_center")

    def _stack_window(self, band: str, sub: pd.DataFrame) -> dict | None:
        """IVW stack for one window of epochs; returns None when no valid rows."""
        valid = sub[np.isfinite(sub["flux"]) & (sub["flux_err"] > 0)]
        if valid.empty:
            return None
        inv_var = 1.0 / valid["flux_err"] ** 2
        f_stack = float((valid["flux"] * inv_var).sum() / inv_var.sum())
        e_stack = float(1.0 / np.sqrt(inv_var.sum()))
        jd_c = float((valid["obsjd"] * inv_var).sum() / inv_var.sum())
        zp = float(valid["zero_point"].median())
        mag, merr = flux_to_ab_mag(f_stack, zp, e_stack)
        return dict(
            obsjd_center=jd_c,
            band=band,
            flux_stack=f_stack,
            flux_err_stack=e_stack,
            mag_stack=float(mag),
            mag_err_stack=float(merr) if merr is not None else float("nan"),
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
            Long-format DataFrame with columns:
            obsjd_center, band, flux_stack, flux_err_stack, mag_stack, mag_err_stack, n_epochs.
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
        centers = np.arange(df["obsjd"].min() + window_days / 2, df["obsjd"].max(), step)

        records = []
        for jd_c in centers:
            sub = df[(df["obsjd"] >= jd_c - window_days / 2) & (df["obsjd"] <= jd_c + window_days / 2)]
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
