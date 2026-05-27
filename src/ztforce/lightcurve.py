"""Lightcurve: per-epoch storage, stacking, plotting, I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

_BAND_ORDER = ["g", "r", "i"]
SNT = 3.0  # detection signal-to-noise threshold
SNU = 5.0  # sigma multiplier for upper-limit arrows


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
        from .utils import flux_to_ab_mag

        df = self.df
        if jd_min is not None:
            df = df[df["obsjd"] >= jd_min]
        if jd_max is not None:
            df = df[df["obsjd"] <= jd_max]
        target_bands = bands or self.bands

        records = []
        for band in target_bands:
            sub = df[(df["band"] == band) & df["detection"]]
            sub = sub[np.isfinite(sub["flux"]) & (sub["flux_err"] > 0)]
            if sub.empty:
                continue
            inv_var = 1.0 / sub["flux_err"] ** 2
            f_stack = (sub["flux"] * inv_var).sum() / inv_var.sum()
            e_stack = 1.0 / np.sqrt(inv_var.sum())
            zp = sub["zero_point"].median()
            mag, merr = flux_to_ab_mag(float(f_stack), float(zp), float(e_stack))
            records.append(
                dict(
                    band=band,
                    flux_stack=float(f_stack),
                    flux_err_stack=float(e_stack),
                    mag_stack=float(mag) if mag is not None else float("nan"),
                    mag_err_stack=float(merr) if merr is not None else float("nan"),
                    n_epochs=len(sub),
                )
            )
        return pd.DataFrame(records).set_index("band")

    def rolling_stack(
        self,
        window: float,
        window_unit: str = "days",
        bands: list[str] | None = None,
        step: float | None = None,
    ) -> pd.DataFrame:
        """Rolling IVW stack in a sliding window.

        Returns a long-format DataFrame with columns:
          obsjd_center, band, flux_stack, flux_err_stack, mag_stack, mag_err_stack, n_epochs.
        """
        df = self.df
        if window_unit == "days":
            win = window
        elif window_unit == "years":
            win = window * 365.25
        else:
            raise ValueError(f"Unknown window_unit '{window_unit}'. Use 'days' or 'years'.")

        step = step or (win / 2)
        target_bands = bands or self.bands
        jd_min = df["obsjd"].min()
        jd_max = df["obsjd"].max()
        centers = np.arange(jd_min + win / 2, jd_max, step)

        records = []
        for jd_c in centers:
            sub = self.stack(jd_min=jd_c - win / 2, jd_max=jd_c + win / 2, bands=target_bands)
            for band, row in sub.iterrows():
                records.append({"obsjd_center": jd_c, "band": band, **row.to_dict()})
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
        df = t.to_pandas()
        for _, row in df.iterrows():
            lc._rows.append(row.to_dict())
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
