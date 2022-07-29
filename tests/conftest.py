import os
import pytest
import shutil
import errno
from arcwright.arcwright import ArcFAI, ArcModule


def copyFile(src, dst):
    try:
        shutil.copytree(src, dst)
    # Depend what you need here to catch the problem
    except OSError as exc:
        # File already exist
        if exc.errno == errno.EEXIST:
            shutil.copy(src, dst)
        # The dirtory does not exist
        if exc.errno == errno.ENOENT:
            shutil.copy(src, dst)
        else:
            raise


@pytest.fixture
def default_arcfai():
    return ArcFAI()


@pytest.fixture
def default_arcmodule():
    return ArcModule()


@pytest.fixture
def data_directory(tmp_path, request):
    """
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        shutil.copytree(test_dir, tmp_path)

    return tmp_path
