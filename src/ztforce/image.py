"""ZTFImage: FITS loading, WCS, and pixel/sky coordinate transforms."""

from __future__ import annotations

import warnings

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

from .config import ZTForceConfig
from .exceptions import WCSError

# e-/ADU per coadded frame for ZTF deep stacks (Bellm et al. 2019, PASP, 131, 018002)
_ZTF_GAIN_PER_FRAME = 5.8


class ZTFImage:
    """Lazy-loading wrapper around a single ZTF science FITS image."""

    def __init__(self, fits_fpath: str, band: str, config: ZTForceConfig) -> None:
        self._fpath = fits_fpath
        self.band = band
        self._config = config
        self._header: fits.Header | None = None
        self._data: np.ndarray | None = None
        self._wcs: WCS | None = None
        self._nan_mask: np.ndarray | None = None

    # ── raw data ─────────────────────────────────────────────────────────────

    @property
    def header(self) -> fits.Header:
        """FITS primary header."""
        if self._header is None:
            self._load_fits()
        return self._header  # type: ignore[return-value]

    @property
    def data(self) -> np.ndarray:
        """Image array as a native-endian float64."""
        if self._data is None:
            self._load_fits()
        return self._data  # type: ignore[return-value]

    def _load_fits(self) -> None:
        with fits.open(self._fpath) as hdul:
            hdr = hdul[0].header
            # Astropy WCS requires RADESYS; older ZTF headers use RADECSYS
            if "RADECSYS" in hdr and "RADESYS" not in hdr:
                hdr.rename_keyword("RADECSYS", "RADESYS")
            elif "RADECSYS" in hdr:
                del hdr["RADECSYS"]
            self._header = hdr
            raw = hdul[0].data
            self._data = np.ascontiguousarray(raw, dtype=np.float64)

    @property
    def cutout_origin(self) -> tuple[float, float]:
        """Pixel offset (x0, y0) of this cutout's origin within the full quadrant.

        IRSA IBE cutouts include LTV1/LTV2 header keywords following the IRAF
        convention where LTV is the negative of the cutout's 0-indexed starting
        pixel: x_full = x_cutout - LTV1.  Returns (0.0, 0.0) for full images.
        """
        ltv1 = float(self.header.get("LTV1", 0.0))
        ltv2 = float(self.header.get("LTV2", 0.0))
        return -ltv1, -ltv2

    # ── derived scalar properties ─────────────────────────────────────────────

    @property
    def gain(self) -> float:
        """Effective gain in e-/ADU."""
        hdr = self.header
        if "GAIN" in hdr:
            return float(hdr["GAIN"])
        if "NFRAMES" in hdr:
            return _ZTF_GAIN_PER_FRAME * float(hdr["NFRAMES"])
        return self._config.default_gain

    @property
    def fwhm(self) -> float:
        """Median PSF FWHM in pixels from header."""
        hdr = self.header
        if "MEDFWHM" in hdr:
            return float(hdr["MEDFWHM"])
        return float(hdr["SEEING"])

    @property
    def zero_point(self) -> float:
        """AB photometric zero-point from header (MAGZP).

        Calibrated against PanSTARRS DR1 by the ZTF pipeline
        (Masci et al. 2019, PASP, 131, 018003).
        """
        return float(self.header["MAGZP"])

    @property
    def obs_jd(self) -> float:
        """Observation Julian date."""
        return float(self.header["OBSJD"])

    @property
    def mag_limit(self) -> float | None:
        """5-sigma limiting magnitude from header, if present."""
        v = self.header.get("MAGLIM")
        return float(v) if v is not None else None

    # ── WCS ──────────────────────────────────────────────────────────────────

    @property
    def wcs(self) -> WCS:
        """Astropy WCS built from the image header."""
        if self._wcs is None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    self._wcs = WCS(self.header)
            except Exception as exc:
                raise WCSError(f"Failed to build WCS for {self._fpath}: {exc}") from exc
        return self._wcs

    def sky_to_pixel(self, coord: SkyCoord) -> tuple[float, float]:
        """Return (x, y) pixel position within this image (cutout-local)."""
        try:
            x, y = self.wcs.world_to_pixel(coord)
            return float(x), float(y)
        except Exception as exc:
            raise WCSError(f"sky_to_pixel failed: {exc}") from exc

    def sky_to_full_quadrant_pixel(self, coord: SkyCoord) -> tuple[float, float]:
        """Return (x, y) in full-quadrant pixel coordinates.

        Required for the spatially-varying DAOPhot PSF model, whose polynomial
        coefficients are indexed to the full quadrant, not the cutout.
        Identical to sky_to_pixel when the image is a full quadrant download.
        """
        x_cut, y_cut = self.sky_to_pixel(coord)
        x0, y0 = self.cutout_origin
        return x_cut + x0, y_cut + y0

    def pixel_to_sky(self, x: float, y: float) -> SkyCoord:
        """Return a SkyCoord for pixel position (x, y)."""
        try:
            return self.wcs.pixel_to_world(x, y)
        except Exception as exc:
            raise WCSError(f"pixel_to_sky failed: {exc}") from exc

    def footprint(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ((ra_min, ra_max), (dec_min, dec_max)) of the image footprint."""
        ny, nx = self.data.shape
        corners = [self.pixel_to_sky(x, y) for x, y in [(0, 0), (nx, 0), (nx, ny), (0, ny)]]
        ras = [c.ra.deg for c in corners]
        decs = [c.dec.deg for c in corners]
        return (min(ras), max(ras)), (min(decs), max(decs))

    # ── masks ─────────────────────────────────────────────────────────────────

    @property
    def nan_mask(self) -> np.ndarray:
        """Boolean mask: True where pixel is NaN."""
        if self._nan_mask is None:
            self._nan_mask = ~np.isfinite(self.data)
        return self._nan_mask
