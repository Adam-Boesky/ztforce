"""ztforce — forced PSF photometry on ZTF science images."""

from ._version import __version__
from .config import ZTForceConfig, build_config
from .exceptions import (
    ConfigError,
    FITSDownloadError,
    NoImagesFoundError,
    PSFBuildError,
    WCSError,
    ZTForceError,
)
from .lightcurve import Lightcurve
from .pipeline import run_forced_photometry, run_forced_photometry_batch

# Bump when the photometry algorithm changes in a way that makes existing
# cached lightcurves stale (e.g. new background estimator, PSF fitting change).
_PHOTOMETRY_VERSION = "1"

__all__ = [
    "__version__",
    "ZTForceConfig",
    "build_config",
    "Lightcurve",
    "run_forced_photometry",
    "run_forced_photometry_batch",
    "ZTForceError",
    "ConfigError",
    "FITSDownloadError",
    "NoImagesFoundError",
    "PSFBuildError",
    "WCSError",
]
