# Bump when the photometry algorithm changes in a way that makes existing
# cached lightcurves stale (e.g. new background estimator, PSF fitting change).
_PHOTOMETRY_VERSION = "8"

# CRPIX of a full ZTF CCD-quadrant science image (archive metadata crpix1/crpix2).
# IBE cutouts carry no LTV keywords; they shift CRPIX instead, so a cutout's offset
# within its quadrant is this minus the cutout's own CRPIX.
ZTF_QUADRANT_CRPIX: tuple[float, float] = (1536.5, 1540.5)

# Default IRSA IBE cutout size for ZTF science image downloads.
DEFAULT_CUTOUT_SIZE_ARCMIN: float = 2.0

# Epoch quality flags: a bitmask stored in the lightcurve ``flags`` column; 0 = good.
# Flagged epochs are never detections and are left out of stacks.
FLAG_FIT_FAILED = 1  # target off the image edge or in a NaN region
FLAG_PROCESSING_ERROR = 2  # the image or PSF could not be read or fitted
# Quality cuts from the ZFPS user guide (Masci et al. 2023, section 6.1).
FLAG_BAD_CALIBRATION = 4  # INFOBITS >= 2**25: bad photometric calibration
FLAG_NOISY_IMAGE = 8  # robust per-pixel noise in the science image > 25 DN
FLAG_BAD_SEEING = 16  # seeing FWHM > 4 arcsec
# Epochs listed in the archive metadata but not measured.
FLAG_UNAVAILABLE = 32  # IRSA does not serve the file (404/410/401/403); not retried automatically
FLAG_DOWNLOAD_FAILED = 64  # download kept failing (timeouts, 5xx); retried on the next run
FLAG_SATURATED = 128  # a pixel under the PSF is at or above the image's SATURATE level

BAD_CALIBRATION_INFOBITS = 2**25
MAX_SCISIGPIX_DN = 25.0
MAX_SEEING_ARCSEC = 4.0

# PSF-fitting radius in pixels, as ZTF's own DAOPhot/AllStar run sets it from the image
# FWHM: FI = min(7, max[3, FW]) (Masci et al. 2017, PASP 129, 014002, eq. 45).  Pixels
# beyond it carry no weight in the fit; inside it the weight tapers to zero at the edge.
PSF_FIT_RADIUS_MIN_PX = 3.0
PSF_FIT_RADIUS_MAX_PX = 7.0

# Sky annulus for the forced fit, in pixels beyond the PSF radius: just outside the
# region the PSF model covers, so the star's own wings are not taken as sky.  A closer
# annulus (2-4 FWHM) biased blank-sky fluxes positive by ~0.4 sigma.
SKY_ANNULUS_GAP_PX = 1
SKY_ANNULUS_WIDTH_PX = 8
