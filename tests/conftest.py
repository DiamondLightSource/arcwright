import os
import pytest
from distutils import dir_util
from arcwright.arcwright import ArcFAI, ArcModule


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
        dir_util.copy_tree(test_dir, str(tmp_path))

    return tmp_path
