import pytest
from arcwright.arcwright import ArcFAI, ArcModule


@pytest.fixture
def default_arcfai():
    return ArcFAI()


@pytest.fixture
def default_arcmodule():
    return ArcModule()
