"""
Test the ArcModule class
"""
from typing import OrderedDict
from pyFAI.detectors import Detector


def test_constructor(default_arcmodule):
    """
    test it constructs
    """
    _ = default_arcmodule
    assert True


def test_is_Detector(default_arcmodule):
    """
    test that the armcodule object ineherits from pyFAI Detector
    """
    assert isinstance(default_arcmodule, Detector)


def test_arcmodule_aliaases(default_arcmodule):
    """
    test the aliases
    """
    assert "XpdfArcModule" in default_arcmodule.aliases


def test_get_config(default_arcmodule):
    """
    test the get_config method returns an ordered dict with the correct keys
    """
    result = default_arcmodule.get_config()
    assert isinstance(result, OrderedDict)
    assert all(x in ["pixel1", "pixel2"] for x in result.keys())
