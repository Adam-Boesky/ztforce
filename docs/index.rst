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
   lcs["g"].plot()
   lcs["g"].stack()

See :doc:`Notebooks <notebooks>` for a worked example.

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

How it works
------------

For each ZTF science image that covers the target position, ztforce:

1. Downloads a small FITS cutout centred on the target from the IRSA IBE cutout service.
2. Subtracts a sigma-clipped sky level estimated from an annulus around the source.
3. Fits the source amplitude using the matched-filter estimator with the per-image
   DAOPhot PSF sidecar (``sciimgdao.psf``) that IRSA delivers alongside every science image.
4. Converts to AB magnitudes using the ``MAGZP`` zero-point already calibrated against
   PanSTARRS DR2 by the ZTF pipeline — no external catalog required.

Downloaded cutouts and the resulting lightcurves are cached on disk; repeated calls for
the same position return immediately.

Contributing
------------

.. code-block:: console

   >> git clone https://github.com/Adam-Boesky/ztforce && cd ztforce
   >> pip install -e '.[dev]'
   >> pre-commit install

Run tests with ``pytest`` (network tests are excluded by default; add ``-m network`` to
include them).

.. toctree::
   :hidden:

   Home page <self>
   API Reference <autoapi/index>
   Notebooks <notebooks>
