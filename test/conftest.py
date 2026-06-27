import os
import pytest


@pytest.fixture(scope="function")
def datadir():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    yield os.path.join(test_dir, "data")


@pytest.fixture(scope="function")
def datafile(datadir):
    """Return a callable that resolves a path inside the test data directory.

    If the requested file is not present the test is skipped with a clear
    message. Some upstream ExodusII reference files (e.g. the joined serial
    ``noh.exo`` and ``edges.base.exo``) are not redistributed with this fork,
    so tests that depend on them skip rather than fail on a fresh checkout.
    """

    def _resolve(name):
        path = os.path.join(datadir, name)
        if not os.path.exists(path):
            pytest.skip(f"required test data file not found: {name}")
        return path

    return _resolve
