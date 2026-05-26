"""Custom exception classes for ztforce."""


class ZTForceError(Exception):
    """Base class for all ztforce exceptions."""


class ConfigError(ZTForceError):
    """Bad or missing credentials / configuration."""


class FITSDownloadError(ZTForceError):
    """FITS file download failed after maximum retries."""


class NoImagesFoundError(ZTForceError):
    """No ZTF science images cover the requested position."""


class PSFBuildError(ZTForceError):
    """PSF model could not be built or parsed for an image."""


class WCSError(ZTForceError):
    """WCS creation or coordinate transformation failed."""
