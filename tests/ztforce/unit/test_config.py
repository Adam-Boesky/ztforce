"""Tests for ztforce.config."""

import pytest


def _clean_env(monkeypatch):
    monkeypatch.delenv("ZTFORCE_IRSA_USER", raising=False)
    monkeypatch.delenv("ZTFORCE_IRSA_PASS", raising=False)


# ── Resolution priority ───────────────────────────────────────────────────────


def test_direct_params_highest_priority(monkeypatch, tmp_path):
    """Direct keyword arguments override every other source."""
    monkeypatch.setenv("ZTFORCE_IRSA_USER", "envuser")
    monkeypatch.setenv("ZTFORCE_IRSA_PASS", "envpass")
    from ztforce.config import build_config

    cfg = build_config(irsa_user="direct", irsa_pass="pw", config_path=tmp_path / "x.toml")
    assert cfg.irsa_user == "direct"
    assert cfg.irsa_pass == "pw"


def test_env_vars_second_priority(monkeypatch, tmp_path):
    """Environment variables are used when no direct params are given."""
    monkeypatch.setenv("ZTFORCE_IRSA_USER", "envuser")
    monkeypatch.setenv("ZTFORCE_IRSA_PASS", "envpass")
    from ztforce.config import build_config

    cfg = build_config(config_path=tmp_path / "x.toml")
    assert cfg.irsa_user == "envuser"
    assert cfg.irsa_pass == "envpass"


def test_toml_file_third_priority(monkeypatch, tmp_path):
    """TOML config file is used when env vars are absent."""
    _clean_env(monkeypatch)
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[credentials]\nirsa_user = 'tomluser'\nirsa_pass = 'tomlpass'\n")
    from ztforce.config import build_config

    cfg = build_config(config_path=cfg_path)
    assert cfg.irsa_user == "tomluser"
    assert cfg.irsa_pass == "tomlpass"


def test_config_error_when_no_source(monkeypatch, tmp_path):
    """ConfigError is raised when no credential source provides both values."""
    _clean_env(monkeypatch)
    from ztforce.config import build_config
    from ztforce.exceptions import ConfigError

    with pytest.raises(ConfigError, match="IRSA credentials not found"):
        build_config(config_path=tmp_path / "x.toml")


# ── Defaults and overrides ────────────────────────────────────────────────────


def test_default_photometry_params(monkeypatch, tmp_path):
    """ZTForceConfig is populated with sane defaults when not overridden."""
    monkeypatch.setenv("ZTFORCE_IRSA_USER", "u")
    monkeypatch.setenv("ZTFORCE_IRSA_PASS", "p")
    from ztforce.config import ZTForceConfig, build_config

    cfg = build_config(config_path=tmp_path / "x.toml")
    defaults = ZTForceConfig(irsa_user="u", irsa_pass="p")
    assert cfg.sep_bw == defaults.sep_bw
    assert cfg.max_retries == defaults.max_retries
    assert cfg.default_gain == defaults.default_gain


def test_photometry_overrides_via_kwargs(monkeypatch, tmp_path):
    """Photometry parameters can be overridden via build_config kwargs."""
    monkeypatch.setenv("ZTFORCE_IRSA_USER", "u")
    monkeypatch.setenv("ZTFORCE_IRSA_PASS", "p")
    from ztforce.config import build_config

    cfg = build_config(config_path=tmp_path / "x.toml", max_retries=10, default_gain=5.0)
    assert cfg.max_retries == 10
    assert cfg.default_gain == 5.0


def test_photometry_overrides_via_toml(monkeypatch, tmp_path):
    """Photometry parameters from [photometry] TOML section are applied."""
    _clean_env(monkeypatch)
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[credentials]\nirsa_user='u'\nirsa_pass='p'\n" "[photometry]\nmax_retries=7\n")
    from ztforce.config import build_config

    cfg = build_config(config_path=cfg_path)
    assert cfg.max_retries == 7
