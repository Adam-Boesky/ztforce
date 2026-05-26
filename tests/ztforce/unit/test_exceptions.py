"""Tests for ztforce.exceptions."""

import pytest


def test_all_exceptions_subclass_ztforce_error():
    """Every domain exception is catchable as ZTForceError."""
    from ztforce.exceptions import (
        ConfigError,
        FITSDownloadError,
        NoImagesFoundError,
        PSFBuildError,
        WCSError,
        ZTForceError,
    )

    leaves = [ConfigError, FITSDownloadError, NoImagesFoundError, PSFBuildError, WCSError]
    for cls in leaves:
        assert issubclass(cls, ZTForceError), f"{cls.__name__} not a subclass of ZTForceError"


def test_exceptions_can_be_raised_and_caught_as_base():
    """Instances of leaf exceptions are catchable via the base class."""
    from ztforce.exceptions import ConfigError, ZTForceError

    with pytest.raises(ZTForceError):
        raise ConfigError("bad credentials")


def test_exceptions_carry_message():
    """Exception message is preserved."""
    from ztforce.exceptions import PSFBuildError

    msg = "no valid stars"
    exc = PSFBuildError(msg)
    assert str(exc) == msg
