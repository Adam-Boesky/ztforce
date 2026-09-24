"""Custom exception classes for ztforce."""


class ZTForceError(Exception):
    """Base class for all ztforce exceptions."""


class ConfigError(ZTForceError):
    """Bad or missing credentials / configuration."""


class FITSDownloadError(ZTForceError):
    """FITS file download failed after maximum retries."""


class ProductUnavailableError(FITSDownloadError):
    """IRSA does not serve the requested file (HTTP 401/403/404/410); retrying won't help."""

    def __init__(self, url: str, status: int) -> None:
        super().__init__(f"HTTP {status} for {url}")
        self.url = url
        self.status = status


class NoImagesFoundError(ZTForceError):
    """No ZTF science images cover the requested position."""


class PSFBuildError(ZTForceError):
    """PSF model could not be built or parsed for an image."""


class WCSError(ZTForceError):
    """WCS creation or coordinate transformation failed."""
