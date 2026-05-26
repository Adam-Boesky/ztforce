"""Credential resolution and photometry hyperparameters."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from .exceptions import ConfigError

# Environment variable names for credentials
_ENV_IRSA_USER = "ZTFORCE_IRSA_USER"
_ENV_IRSA_PASS = "ZTFORCE_IRSA_PASS"

# Default config file location
_DEFAULT_CONFIG_PATH = Path.home() / ".ztforce" / "config.toml"


@dataclass
class ZTForceConfig:
    """All runtime configuration for a ztforce session."""

    # Credentials
    irsa_user: str = ""
    irsa_pass: str = ""

    # SEP background / extraction
    sep_bw: int = 64
    sep_bh: int = 64
    sep_fw: int = 3
    sep_fh: int = 3

    # PSF sizing (cutout = nearest_odd_int(FWHM * factor))
    psf_cutout_fwhm_factor: float = 14.0

    # IRSA IBE cutout size for science image downloads
    cutout_size_arcmin: float = 15.0

    # Retry / download
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_jitter: float = 0.5

    # Gain fallback when GAIN header is absent
    default_gain: float = 6.2


def _load_toml(path: Path) -> dict:
    """Load a TOML file; return {} if absent or unreadable."""
    if not path.exists():
        return {}
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_config(
    irsa_user: str | None = None,
    irsa_pass: str | None = None,
    config_path: str | Path | None = None,
    **overrides,
) -> ZTForceConfig:
    """Build a ZTForceConfig by resolving credentials in priority order.

    Resolution order (highest first):
    1. Direct parameters passed here
    2. Environment variables ZTFORCE_IRSA_USER / ZTFORCE_IRSA_PASS
    3. ~/.ztforce/config.toml  [credentials] section

    Raises ConfigError listing all sources tried if credentials are not found.
    """
    cfg_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    toml_data = _load_toml(cfg_path)

    def resolve(key_direct, env_var, toml_key):
        if key_direct:
            return key_direct
        if env_val := os.environ.get(env_var):
            return env_val
        if toml_val := toml_data.get("credentials", {}).get(toml_key):
            return toml_val
        return None

    resolved_user = resolve(irsa_user, _ENV_IRSA_USER, "irsa_user")
    resolved_pass = resolve(irsa_pass, _ENV_IRSA_PASS, "irsa_pass")

    if not resolved_user or not resolved_pass:
        tried = [
            "  direct parameters (irsa_user/irsa_pass)",
            f"  env vars: {_ENV_IRSA_USER}, {_ENV_IRSA_PASS}",
            f"  config file: {cfg_path}  ([credentials] irsa_user/irsa_pass)",
        ]
        raise ConfigError("IRSA credentials not found. Tried:\n" + "\n".join(tried))

    # Photometry overrides from config.toml [photometry] section
    phot_section = toml_data.get("photometry", {})
    cfg = ZTForceConfig(irsa_user=resolved_user, irsa_pass=resolved_pass)
    for k, v in {**phot_section, **overrides}.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
