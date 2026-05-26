import ztforce


def test_version():
    """Check to see that we can get the package version"""
    assert ztforce.__version__ is not None
