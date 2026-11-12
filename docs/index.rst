.. ztforce documentation main file.

ztforce
=======

**ztforce** measures flux at a fixed sky position in every available ZTF science image —
even when the source is below the detection threshold — and returns a calibrated
AB-magnitude lightcurve.  It uses the DAOPhot PSF sidecars delivered alongside every
ZTF science image, so no EPSF-building or external catalog query is needed.

.. code-block:: python

   from ztforce import run_forced_photometry

   lcs = run_forced_photometry(ra=210.08, dec=-6.88, bands=["g", "r"])
   lcs["g"].df       # pandas DataFrame of all epochs
   lcs["g"].stack()  # inverse-variance weighted stack of all good epochs

See :doc:`Notebooks <notebooks>` for a worked example and the
:doc:`API Reference <autoapi/index>` for every option.

Installation
------------

.. code-block:: console

   >> pip install ztforce

**Python 3.10 – 3.13** is supported.

Credentials
^^^^^^^^^^^

ztforce downloads ZTF science images from IRSA.
Register for a free account at `irsa.ipac.caltech.edu <https://irsa.ipac.caltech.edu>`_,
then supply your credentials in one of three ways (highest priority first):

**1. Direct argument:**

.. code-block:: python

   from ztforce import build_config
   config = build_config(irsa_user="you@example.com", irsa_pass="secret")
   lcs = run_forced_photometry(..., config=config)

**2. Environment variables** (recommended for scripts and CI):

.. code-block:: bash

   export ZTFORCE_IRSA_USER=you@example.com
   export ZTFORCE_IRSA_PASS=secret

**3. Config file** at ``~/.ztforce/config.toml``:

.. code-block:: toml

   [credentials]
   irsa_user = "you@example.com"
   irsa_pass = "secret"

Usage
-----

Single target
^^^^^^^^^^^^^

.. code-block:: python

   from ztforce import run_forced_photometry

   # Measure flux at a fixed position across all ZTF g- and r-band epochs.
   # A tqdm progress bar tracks downloads and PSF fitting; results are cached
   # on disk so repeated calls return instantly.
   lcs = run_forced_photometry(ra=210.08, dec=-6.88, bands=["g", "r"])

   lcs["g"].df                           # pandas DataFrame of all epochs
   lcs["g"].stack()                      # inverse-variance weighted stack of all good epochs
   lcs["g"].rolling_stack(window=30.0)   # 30-day stacks (window_unit="days" | "years" | "images")
   lcs["g"].save("my_source_g.ecsv")     # save to ECSV

Batch processing
^^^^^^^^^^^^^^^^

.. code-block:: python

   from astropy.coordinates import SkyCoord
   from ztforce import run_forced_photometry_batch

   targets = SkyCoord(ra=[210.08, 130.13], dec=[-6.88, 19.70], unit="deg")

   # Processes targets in parallel; the metadata query, the download pool and
   # the PSF sidecars are shared across all workers.
   results = run_forced_photometry_batch(targets, bands=["g", "r"], n_workers=4)

   results[0]["g"].stack()  # stacked photometry for the first target, g-band

What each epoch holds
^^^^^^^^^^^^^^^^^^^^^

Every exposure whose footprint contains the target appears as a row, with its ZTF
``field``, ``ccdid`` and ``qid``:

- ``flux``, ``flux_err``: instrumental flux (DN) on the epoch's ``zero_point``, kept at any S/N.
- ``mag``, ``mag_err``: AB magnitudes (ZTF calibration, zero colour, like ZTF's own
  catalogs), **only for detections** (S/N ≥ 3).  A good non-detection has a 5σ
  ``upper_limit`` at the target instead.
- ``flags``: a bitmask, 0 = good.  Flagged epochs are never detections or stacked:

===== ==================================================================
Bit   Meaning
===== ==================================================================
1     fit failed (target off the image or in a NaN region)
2     image or PSF could not be read
4     bad photometric calibration (archive ``infobits`` ≥ 2\ :sup:`25`)
8     noisy image (robust pixel noise > 25 DN)
16    seeing > 4″
32    IRSA does not serve the file
64    download kept failing (retried on the next run)
128   saturated pixel under the PSF
===== ==================================================================

Epochs the archive metadata already marks as bad (bits 4, 16) are not downloaded unless
``measure_flagged=True``.  Files IRSA does not serve (some exposures listed in its
metadata have no files) are not re-requested unless ``retry_unavailable=True``.  A warning
summarises epochs that could not be measured.  If the IRSA metadata search itself keeps
failing (an outage, not empty sky), :func:`~ztforce.run_forced_photometry` raises
:class:`~ztforce.exceptions.MetadataQueryError`, and
:func:`~ztforce.run_forced_photometry_batch` puts that error in the failed target's place,
with a warning, and completes the rest.

Stacking and uncertainties
^^^^^^^^^^^^^^^^^^^^^^^^^^

Quality cuts and stacking follow the
`ZFPS user guide <https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_zfps_userguide.pdf>`_:
stacks combine all good epochs in flux, rescaled to a common zero point and
inverse-variance averaged, and are reported as a 5σ upper limit below S/N 3.

Uncertainties are statistical (sky + Poisson).  They describe the scatter of faint and
blank-sky measurements well, but omit calibration and PSF systematics, so they are
**underestimated for bright sources** (ZTF's own catalogs carry a ~1% floor).  The fit is
on science images, not difference images, so light from an extended host can vary with
seeing.

Caching
^^^^^^^

Lightcurves are cached per position and band under ``~/.ztforce/cache`` (or ``data_dir``).
A cached lightcurve does not pick up epochs taken after it was built; ztforce warns once
it is over 30 days old, and ``force_recompute=True`` rebuilds it.

How it works
------------

For each ZTF science image that covers the target position, ztforce:

1. Downloads a small FITS cutout centred on the target from the IRSA IBE cutout service.
2. Subtracts a sigma-clipped sky level estimated from an annulus just outside the PSF
   footprint.
3. Fits one amplitude at the fixed position with the matched-filter estimator, using the
   image's own DAOPhot PSF (``sciimgdao.psf``), rendered as DAOPhot does including its
   slightly negative wings.  The fit is weighted towards the core with DAOPhot's radial
   taper inside ZTF's fitting radius, min(7, max(3, FWHM)) px
   (`Masci et al. 2017 <https://arxiv.org/abs/1608.01733>`_, eq. 45), so light from
   neighbours or a host galaxy in the outer PSF box is not counted.
   ``build_config(psf_fit_taper=False)`` fits the whole box instead.
4. Converts to AB magnitudes using the ``MAGZP`` zero-point already calibrated against
   Pan-STARRS1 by the ZTF pipeline — no external catalog required.

Downloaded cutouts are deleted after fitting; the resulting lightcurves are cached on
disk, so repeated calls for the same position return immediately.

Accuracy
^^^^^^^^

For isolated stars, ztforce reproduces the fluxes in ZTF's own per-image PSF catalogs
(``psfcat``, same pixels and PSF) to within ~1%, with ~1% scatter and reduced χ² ≈ 1, and
blank sky comes out consistent with zero.

Related services
----------------

Several official ZTF services offer complementary photometry:

- `ZTF Forced Photometry Service (ZFPS) <https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_forced_photometry.pdf>`_
  (`Masci et al. 2023 <https://arxiv.org/abs/2305.16279>`_): forced PSF photometry on ZTF
  **difference** images at user positions.  ztforce works on science images, so it avoids
  subtraction artifacts, needs no request queue, and runs locally.
- `ZTF DR lightcurves (IRSA) <https://irsa.ipac.caltech.edu/docs/program_interface/ztf_lightcurve_api.html>`_:
  catalog lightcurves for sources detected in ZTF data releases.  ztforce also covers
  transients and sub-threshold sources that are in no catalog, at any position.
- `ZuberCal <https://irsa.ipac.caltech.edu/data/ZTF/zubercal/overview.pdf>`_: ubercalibrated
  PSF photometry for PS1-matched ZTF sources only.

ztforce is designed for cases where the source may not appear in any existing catalog —
supernovae, kilonovae, flares, or any transient — and you need calibrated flux
measurements at a fixed sky position across every available epoch, including
non-detections.

Citation
--------

If you find ztforce useful for your work, please cite the Zenodo archive
(`doi:10.5281/zenodo.20434974 <https://doi.org/10.5281/zenodo.20434974>`_, which resolves
to the latest release).  See the README for a BibTeX entry.

Contributing
------------

.. code-block:: console

   >> git clone https://github.com/Adam-Boesky/ztforce && cd ztforce
   >> pip install -e '.[dev]'
   >> pre-commit install

Run tests with ``pytest`` (network tests are excluded by default; add ``-m network`` to
include them).  See
`CONTRIBUTING.md <https://github.com/Adam-Boesky/ztforce/blob/main/CONTRIBUTING.md>`_
for details.

.. toctree::
   :hidden:

   Home page <self>
   API Reference <autoapi/index>
   Notebooks <notebooks>
