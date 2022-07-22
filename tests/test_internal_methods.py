"""
Test the internally used methods
"""

from pytest import approx
import pytest
import numpy as np
from arcwright.arcwright import ArcFAI


@pytest.mark.parametrize(
    "two_theta, points", [(0, 1.0 / 3), (10, 1.0 / 3), (20, 0.61516774), (181, 0)]
)
def test_get_pts_per_deg(default_arcfai, two_theta, points):
    """
    test the behaviour of the pts_per_deg function
    """
    assert default_arcfai.get_pts_per_deg(two_theta) == approx(points)


def test_number_of_images_increases():
    """
    test adding images makes self-consistent ArcFAI().goniometers
    """
    arcfai = ArcFAI()
    data = np.ones((10, 10))
    for x in range(3):
        arcfai.add_imgs_to_goniometer_refinements([data], [x])
        assert all(
            len(gonio.single_geometries) == x + 1
            for gonio in arcfai.goniometers.values()
        )


def test_number_of_images_increases_same_angle():
    """
    test adding images makes self-consistent ArcFAI().goniometers
    """
    arcfai = ArcFAI()
    data = np.ones((10, 10))
    for x in range(3):
        arcfai.add_imgs_to_goniometer_refinements([data], [0.0])
        assert all(
            len(gonio.single_geometries) == x + 1
            for gonio in arcfai.goniometers.values()
        )


def test_gonios_param_to_string(capfd, default_arcfai):
    """
    test the gonios_param_to_string prints a bunch of lines
    """
    default_arcfai.gonios_param_to_string()
    out, err = capfd.readouterr()

    assert err == ""
    assert len(out.split("\n")) == default_arcfai.n_modules + 2
    assert all(x in out for x in default_arcfai.module_names)


def test_multigonios_param_to_string(capfd, default_arcfai):
    """
    test the multigonios_param_to_string prints a bunch of lines
    """
    default_arcfai.multigonio_param_to_string()
    out, err = capfd.readouterr()

    assert err == ""
    assert len(out.split("\n")) == default_arcfai.n_modules + 4
    assert all(x in out for x in default_arcfai.module_names)
    assert "scale:" in out
    assert "nrj:" in out


def test_multigonio_param_bounds_to_string(capfd, default_arcfai):
    """
    test that test_multigonio_param_bounds_to_string does nothing
    """
    result = default_arcfai.multigonio_param_bounds_to_string()
    assert result is None
    out, err = capfd.readouterr()
    assert err == ""
    assert out == ""


def test_residu2():
    """
    test the residu2 function returns some expected values, and errors out if you call it
    with uncalibrated geometries
    """
    arcfai = ArcFAI()
    assert arcfai.residu2(arcfai.param) == approx(0)
    data = np.ones((6604, 768))
    arcfai.add_imgs_to_goniometer_refinements([data], [0])
    with pytest.raises(TypeError):
        _ = arcfai.residu2(arcfai.param)


def test_chi2():
    """
    tests the chi2 wrapping of residu2
    """
    arcfai = ArcFAI()
    assert arcfai.chi2() == approx(0)


def test_update_gonio_params(capfd):
    """
    tests the update_gonio_params behaves as expected
    """
    arcfai = ArcFAI()
    arcfai.param = np.ones_like(arcfai.param)
    arcfai.update_gonio_params()
    arcfai.gonios_param_to_string()

    out, err = capfd.readouterr()
    assert err == ""
    assert out.count("1.0") == 7 * arcfai.n_modules


def test_update_gonio_params_2(capfd):
    """
    tests the update_gonio_params behaves as expected
    """
    arcfai = ArcFAI()
    arcfai.param = np.ones_like(arcfai.param)
    arcfai.update_gonio_params()
    arcfai.multigonio_param_to_string()

    out, err = capfd.readouterr()
    assert err == ""
    assert out.count("1.0") == 2 + 5 * arcfai.n_modules


@pytest.mark.parametrize("this_shape", [(10, 10), (3000, 300), (6604, 768)])
def test_solidAngleArray_basic(this_shape):
    """
    test the solidAngleArray method
    TODO maybe this is a garbage test
    """
    arcfai = ArcFAI()
    data = np.ones(this_shape)
    arcfai.add_imgs_to_goniometer_refinements([data], [0])
    result = arcfai.solidAngleArray()
    assert isinstance(result, list)
    assert result[0].shape == this_shape


def test_solidAngleArray_relative():
    """
    test the solidAngleArray method with absolute values
    """
    arcfai = ArcFAI()
    data = np.ones((6604, 768))
    arcfai.add_imgs_to_goniometer_refinements([data], [0])
    relative = arcfai.solidAngleArray(kwargs={"absolute": False})
    assert np.nanmax(relative[0]) == approx(1, 1e-6)
    # TODO improve this test with a numerical test i.e. calculate the solidangle of some pixels


def test_solidAngleArray_absolute():
    """
    test the solidAngleArray method with absolute values
    """
    arcfai = ArcFAI()
    data = np.ones((6604, 768))
    arcfai.add_imgs_to_goniometer_refinements([data], [0])
    absolute = arcfai.solidAngleArray(kwargs={"absolute": True})
    approximate_solid_angle = arcfai.pixel_size**2 / arcfai.dist
    assert np.nanmax(absolute[0]) == approx(approximate_solid_angle)


def test_solidAngleArray_multiple():
    """
    test the solidAngleArray method with multiple datapoints
    """
    arcfai = ArcFAI()
    data = np.ones((6604, 768))
    arcfai.add_imgs_to_goniometer_refinements([data], [0])
    arcfai.add_imgs_to_goniometer_refinements([data], [1])
    result = arcfai.solidAngleArray()
    assert len(result) == 2


def test_cos_incidence():
    """
    test that the cos_incidence method returns something sensible looking
    """
    arcfai = ArcFAI()
    x = np.arange(256)
    y = np.arange(768)
    YY, XX = np.meshgrid(y, x)
    cos_incidence = arcfai.cos_incidence((XX, YY), tth=0)
    assert np.nanmin(cos_incidence) > 0.99
    assert cos_incidence.shape == YY.shape


def test_image_conversion():
    """
    test the get_img_from_module_imgs and get_module_imgs_from_img methods by converting and
    back-converting some noise
    """
    arcfai = ArcFAI()
    modules = [np.random.random((256, 768)) for _ in range(24)]
    image = arcfai.get_img_from_module_imgs(modules)
    new_modules = arcfai.get_module_imgs_from_img(image)
    assert all(m1 == approx(m2) for m1, m2 in zip(modules, new_modules))
