
# ztforce

Forced PSF photometry on ZTF science images — measures flux at a fixed sky position in every available epoch, even below the detection threshold, producing a calibrated AB-magnitude lightcurve.

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/ztforce?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/ztforce/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/Adam-Boesky/ztforce/smoke-test.yml)](https://github.com/Adam-Boesky/ztforce/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/Adam-Boesky/ztforce/branch/main/graph/badge.svg)](https://codecov.io/gh/Adam-Boesky/ztforce)
[![Read The Docs](https://img.shields.io/readthedocs/ztforce)](https://ztforce.readthedocs.io/)

## Installation

```bash
pip install ztforce
```

**Python 3.10–3.13** is supported.

### Credentials

ztforce downloads ZTF science images from IRSA and requires an IRSA account
(free at [irsa.ipac.caltech.edu](https://irsa.ipac.caltech.edu)). Set your
credentials in one of three ways:

**Environment variables (recommended for scripts/CI):**
```bash
export ZTFORCE_IRSA_USER=your_username
export ZTFORCE_IRSA_PASS=your_password
```

**Config file** at `~/.ztforce/config.toml`:
```toml
[credentials]
irsa_user = "your_username"
irsa_pass = "your_password"
```

**Direct argument:**
```python
from ztforce import build_config
config = build_config(irsa_user="your_username", irsa_pass="your_password")
```

## Quick start

```python
from ztforce import run_forced_photometry

# Measure flux at a fixed position across all ZTF g- and r-band epochs.
# A tqdm progress bar tracks downloads and PSF fitting; results are cached
# on disk so repeated calls return instantly.
lcs = run_forced_photometry(ra=210.08, dec=-6.88, bands=["g", "r"])

lcs["g"].df              # pandas DataFrame of all epochs
lcs["g"].stack()         # inverse-variance weighted stack of detections
lcs["g"].save("my_source_g.ecsv")   # save to ECSV
```

### Batch processing

```python
from astropy.coordinates import SkyCoord
from ztforce import run_forced_photometry_batch

targets = SkyCoord(ra=[210.08, 130.13], dec=[-6.88, 19.70], unit="deg")

# Processes targets in parallel; downloads are shared across all workers.
results = run_forced_photometry_batch(targets, bands=["g", "r"], n_workers=4)

results[0]["g"].stack()  # stacked photometry for first target, g-band
```

## Dev Guide - Getting Started

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

```
>> conda create -n <env_name> python=3.11
>> conda activate <env_name>
```

Once you have created a new environment, you can install this project for local
development using the following commands:

```
>> ./.setup_dev.sh
>> conda install pandoc
```

Notes:
1. `./.setup_dev.sh` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on 
   [pre-commit](https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html)
2. Install `pandoc` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   [Sphinx and Python Notebooks](https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks)
