# Bump when the photometry algorithm changes in a way that makes existing
# cached lightcurves stale (e.g. new background estimator, PSF fitting change).
_PHOTOMETRY_VERSION = "1"

# Default IRSA IBE cutout size for ZTF science image downloads.
DEFAULT_CUTOUT_SIZE_ARCMIN: float = 2.0
