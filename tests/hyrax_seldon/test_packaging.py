import hyrax_seldon


def test_version():
    """Check to see that we can get the package version"""
    assert hyrax_seldon.__version__ is not None
