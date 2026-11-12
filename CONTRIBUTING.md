# Contributing to ztforce

Thanks for your interest in ztforce. Bug reports, feature requests and pull requests are
all welcome at <https://github.com/Adam-Boesky/ztforce/issues>.

## Development setup

```bash
git clone https://github.com/Adam-Boesky/ztforce && cd ztforce
conda create -n ztforce python=3.11 && conda activate ztforce   # or any virtualenv
pip install -e '.[dev]'
pre-commit install
```

`./.setup_dev.sh` does the `pip install` and `pre-commit install` steps for you, and also
installs the documentation requirements.

## Running the tests

```bash
pytest
```

Tests that talk to IRSA are marked `network` and are skipped by default. To run them you
need IRSA credentials (see the README) and `pytest -m network`.

The PSF-fit tests compare ztforce's photometry with ZTF's own `psfcat` catalogs on a frozen
set of cutouts under `tests/data/psfcat_agreement/`. Rebuild that fixture with
`python scripts/build_psfcat_fixture.py` only if you change what it has to contain.

## Code style

`pre-commit` runs ruff (lint + format, line length 110), mypy and the test suite before
each commit; CI runs the same hooks. Direct commits to `main` are blocked by a hook, so
work on a branch and open a pull request.

If you change the photometry in a way that makes existing cached lightcurves stale (a new
background estimator, a different PSF fit, a changed quality cut), bump
`_PHOTOMETRY_VERSION` in `src/ztforce/_constants.py` so cached results are recomputed.

## Documentation

```bash
pip install -r docs/requirements.txt
conda install pandoc          # or brew/apt install pandoc
sphinx-build -T -E -b html docs docs/_build/html
```

The API reference is generated from docstrings by sphinx-autoapi. The notebook in
`docs/pre_executed/` is rendered from its saved outputs and is not executed by the docs
build, so re-run it by hand (it needs IRSA credentials) when its outputs go stale.

## Releasing

Releases are tagged `vX.Y.Z`. Publishing a GitHub release from the tag triggers the
PyPI upload workflow and archives the release on Zenodo.
