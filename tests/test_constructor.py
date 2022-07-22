import pytest
import json
import numpy as np
from pyFAI.containers import Integrate1dResult
from pyFAI.detectors import Detector
from pyFAI.goniometer import GoniometerRefinement, ExtendedTransformation
from arcwright.arcwright import ArcFAI, ArcModule
import random
import string


def test_default_constructor():
    """
    test that ArcFAI constructs without raising an error
    """
    _ = ArcFAI()


def test_goniometers_types(default_arcfai):
    """
    test that all the goniometers are goniometers
    """
    assert all(
        isinstance(x, GoniometerRefinement) for x in default_arcfai.goniometers.values()
    )


def test_modules_types(default_arcfai):
    """
    test that all the modules are Detectors
    """
    assert all(isinstance(x, Detector) for x in default_arcfai.modules.values())
    assert all(isinstance(x, ArcModule) for x in default_arcfai.modules.values())


def test_constructor_arg_dist():
    """
    test that setting the dist works
    """
    dist = random.random()
    arcfai = ArcFAI(dist=dist)
    assert arcfai.dist == pytest.approx(dist)
    assert arcfai.param[::5][:-1] == pytest.approx(dist)


def test_constructor_arg_nrj():
    """
    test that setting the nrj works
    """
    nrj = 100 * random.random()
    arcfai = ArcFAI(nrj=nrj)
    assert arcfai.nrj == pytest.approx(nrj)
    assert arcfai.param[-1] == pytest.approx(nrj)


def test_constructor_arg_n_modules():
    """
    test that setting the n_modules works
    """
    n_modules = random.randint(10, 30)
    arcfai = ArcFAI(n_modules=n_modules)
    assert arcfai.n_modules == n_modules
    assert len(arcfai.module_names) == n_modules
    assert len(arcfai.module_approx_tths) == n_modules
    assert len(arcfai.modules) == n_modules
    assert len(arcfai.goniometers) == n_modules
    assert len(arcfai.param) == 5 * n_modules + 2


def test_constructor_arg_tth_offset():
    """
    test that setting the tth_offset works
    """
    tth_offset = 5 * random.random()
    intermodule = 3.47
    arcfai = ArcFAI(tth_offset=tth_offset, tth_offset_between_modules=intermodule)
    assert arcfai.module_approx_tths[arcfai.module_names[0]] == pytest.approx(
        tth_offset
    )
    assert (arcfai.param[4::5] + tth_offset) / intermodule == pytest.approx(
        np.arange(0, -arcfai.n_modules, -1)
    )


def test_constructor_arg_tth_offset_between_modules():
    """
    test that setting the tth_offset_between_modules works
    """
    intermodule = 5 * random.random()
    arcfai = ArcFAI(tth_offset_between_modules=intermodule)
    assert arcfai.param[4::5] / intermodule == pytest.approx(
        np.arange(0, -arcfai.n_modules, -1)
    )


def generate_json_config(n_modules, module_name_prefix="module"):
    """
    generate a psuedo-noisy json config
    """
    json_blob = {}
    json_blob["wavelength"] = 0.17
    json_blob["param_common"] = {"scale": 1.0000000353397727, "nrj": 76.69}
    json_blob["goniometers"] = {}
    dist = [0.25 + random.random() / 100 for _ in range(n_modules)]
    poni1 = [0.0 + random.random() / 100 for _ in range(n_modules)]
    poni2 = [2e-2 + random.random() / 1000 for _ in range(n_modules)]
    rot1 = [0.0 + random.random() / 100 for _ in range(n_modules)]
    offset = [-5 * i + random.random() / 5 for i in range(n_modules)]
    for i in range(n_modules):
        json_blob["goniometers"][f"{module_name_prefix}_{i}"] = {
            "dist": dist[i],
            "poni1": poni1[i],
            "poni2": poni2[i],
            "rot1": rot1[i],
            "offset": offset[i],
        }
    return json_blob


@pytest.mark.parametrize("n_modules", [24, 10, 3])
def test_constructor_from_json(tmp_path, n_modules):
    """
    test that constructin from json works as expected.
    Note this definitely does not test whether it can _do_ anything once its constructed :)
    (spoilers: it cannot)
    """
    json_blob = generate_json_config(n_modules)
    json_filepath = tmp_path / "test.json"
    with open(json_filepath, "w") as f:
        json.dump(json_blob, f)
    arcfai = ArcFAI(from_json=json_filepath)
    assert len(arcfai.modules) == n_modules


def test_constructor_from_json_with_dict():
    """
    test constructing with the from_json kwarg
    """
    json_blob = generate_json_config(24)
    _ = ArcFAI(from_json=json_blob)


def test_from_json_copes_with_unexpected_names():
    """
    test that the modules in the file are the module_names on the ArcFAI
    TODO think about sorting the names on the import
    """
    random_name = "".join(random.choices(string.ascii_letters, k=16))
    json_blob = generate_json_config(24, random_name)
    arcfai = ArcFAI(from_json=json_blob)
    names_in_json = set(json_blob["goniometers"].keys())
    names_in_arcfai = set(arcfai.module_names)
    assert names_in_json == names_in_arcfai


@pytest.mark.parametrize(
    "energy,allowed", [(40.05, True), (65.40, True), (76.69, True), (100, False)]
)
def test_constructor_from_json_with_nrj(energy, allowed):
    """
    test the from_json method
    """
    arcfai = ArcFAI()
    json_blob = generate_json_config(24)
    if allowed:
        arcfai.from_json(json_blob, energy)
    else:
        with pytest.raises(AssertionError):
            arcfai.from_json(json_blob, energy)


def test_to_json():
    """
    test the to_json method works and outputs a dict which works
    """
    arcfai = ArcFAI()
    result = arcfai.to_json()
    assert isinstance(result, dict)
    arcfai.from_json(result)


def test_to_json_file(tmp_path):
    """
    also test to_json can write a half sensible looking file
    """
    arcfai = ArcFAI()
    json_filepath = tmp_path / "something.json"
    arcfai.to_json(json_filepath)

    arcfai2 = ArcFAI(from_json=json_filepath)
    assert (arcfai.param == arcfai2.param).all()


def test_trans(default_arcfai):
    """
    test the arcfai.trans object looks correct
    """
    assert isinstance(default_arcfai.trans, ExtendedTransformation)
    assert default_arcfai.trans.param_names == (
        "dist",
        "poni1",
        "poni2",
        "rot1",
        "offset",
        "scale",
        "nrj",
    )
    assert default_arcfai.trans.pos_names == ("angle",)
    keys = set(
        [
            "pi",
            "hc",
            "q",
            "dist",
            "poni1",
            "poni2",
            "rot1",
            "offset",
            "scale",
            "nrj",
            "angle",
        ]
    )
    assert keys == default_arcfai.trans.variables.keys()
