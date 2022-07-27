import pytest
from pytest import approx
import numpy as np
import h5py as h5
from pyFAI.containers import Integrate1dResult

from arcwright.arcwright import ArcFAI


def test_simple_integrate1d():
    """
    test that a default ArcFAI can integrate1d some nothing data
    """
    arcfai = ArcFAI()
    data = np.ones((10, 10))
    arcfai.add_imgs_to_goniometer_refinements([data], [0])
    result = arcfai.integrate1d()
    assert isinstance(result, Integrate1dResult)
    assert result.intensity.sum() == approx(14.0)
    assert result.intensity.max() == approx(1.0)
    assert result.intensity.min() == approx(0.0)


def test_full_image_integration():
    """
    test that a default ArcFAI can integrate1d a whole image as expected
    """
    arcfai = ArcFAI()
    data = (np.sin(np.linspace(0, 100, 6604)) ** 2)[:, np.newaxis] * np.ones(768)
    arcfai.add_imgs_to_goniometer_refinements([data], [0])
    result = arcfai.integrate1d()
    assert result.intensity.max() == approx(1, abs=1e-4)
    assert result.intensity.min() == approx(0)


def test_that_one_equals_one():
    """
    test that an image of ones integrates to a plot of ones
    """
    arcfai = ArcFAI()
    data = np.ones((6604, 768))
    arcfai.add_imgs_to_goniometer_refinements([data], [0])
    arcfai.add_imgs_to_goniometer_refinements([data], [1.5])
    result = arcfai.integrate1d(radial_range=(0, 60), npt=6000)
    assert result.intensity.mean() == approx(1.0)


def test_adding_the_same_thing_twice_does_nothing():
    """
    Adding something twice should not impact the integrated intensity
    """
    arcfai = ArcFAI()
    data = (np.sin(np.linspace(0, 100, 6604)) ** 2)[:, np.newaxis] * np.ones(768)
    arcfai.add_imgs_to_goniometer_refinements([data], [0])
    result1 = arcfai.integrate1d()
    arcfai.add_imgs_to_goniometer_refinements([data], [0])
    result2 = arcfai.integrate1d()
    assert result1.intensity == approx(result2.intensity)


def test_sigma_decreases_with_multiple_images():
    """
    Add the same image twice and compare the sigma before and after
    """
    arcfai = ArcFAI()
    data = (np.sin(np.linspace(0, 100, 6604)) ** 2)[:, np.newaxis] * np.ones(768)
    arcfai.add_imgs_to_goniometer_refinements([data], [0])
    result1 = arcfai.integrate1d()
    arcfai.add_imgs_to_goniometer_refinements([data], [0])
    result2 = arcfai.integrate1d()
    assert result1.sigma.sum() > result2.sigma.sum()


def test_sigma_decreases_with_multiple_different_images():
    """
    Add multiple images of ones to check sigma reduces
    """
    arcfai = ArcFAI()
    data = np.ones((6604, 768))
    results = []
    for angle in np.arange(5):
        arcfai.add_imgs_to_goniometer_refinements([data], [angle])
        results.append(arcfai.integrate1d())
    total_sigmas = [x.sigma.sum() for x in results]
    assert all(i > j for i, j in zip(total_sigmas, total_sigmas[1:]))


def test_cossquared_plus_sinsquared_equals_1():
    """
    Adding some squared sine and cosine images should end in something largely flat.
    """
    arcfai = ArcFAI()
    sindata = (np.sin(np.linspace(0, 100, 6604)) ** 2)[:, np.newaxis] * np.ones(768)
    cosdata = (np.cos(np.linspace(0, 100, 6604)) ** 2)[:, np.newaxis] * np.ones(768)
    arcfai.add_imgs_to_goniometer_refinements([sindata], [1e-10])
    arcfai.add_imgs_to_goniometer_refinements([cosdata], [2e-10])
    result = arcfai.integrate1d(radial_range=(0, 40), npt=4000)
    assert result.intensity.mean() == approx(0.5)


def test_integrate1d_masking():
    """
    test applying a mask does something sensible
    """
    arcfai = ArcFAI()
    data = np.ones((6604, 768))
    arcfai.add_imgs_to_goniometer_refinements([data], [0])
    unmasked = arcfai.integrate1d()
    masked = arcfai.integrate1d(mask=data)
    assert masked.intensity.sum() == approx(0)
    assert unmasked.intensity.sum() > masked.intensity.sum()


def test_integrate1d_masking_granular():
    """
    test applying a mask does something sensible with small features
    """
    arcfai = ArcFAI()
    data = np.ones((6604, 768))
    data[3000][300] = 1e10
    arcfai.add_imgs_to_goniometer_refinements([data], [0])
    unmasked = arcfai.integrate1d()
    masked = arcfai.integrate1d(mask=data > 2)
    assert unmasked.intensity.max() > 2
    assert masked.intensity.max() < 2


def test_integrate1d_flat():
    """
    appears to be a pyfai thing - phil has seen it work in jupyter (diamond 3.7 probs)
    confirmed, does something different with module load python/3.7
    """
    data_ramp = np.linspace(0, 100, 6604)[:, np.newaxis] * np.ones(768)
    arcfai = ArcFAI()
    arcfai.add_imgs_to_goniometer_refinements([data_ramp], [0])
    unflattered = arcfai.integrate1d()
    flattered = arcfai.integrate1d(flat=data_ramp)
    difference = flattered.intensity - unflattered.intensity

    assert unflattered.intensity != approx(flattered.intensity)
    assert (np.abs(difference) > 0).sum() > 4500


@pytest.mark.filterwarnings("ignore:Exception")
@pytest.mark.parametrize("points", [12, 1540, 9000, 1.6, "nope"])
def test_integrate1d_number_points(points):
    """
    test passing a number of points into the integrate1d method
    """
    arcfai = ArcFAI()
    data = np.ones((6604, 768))
    arcfai.add_imgs_to_goniometer_refinements([data], [0])

    if not isinstance(points, int):
        with pytest.raises(Exception):
            _ = arcfai.integrate1d(npt=points)
    else:
        results = arcfai.integrate1d(npt=points)
        assert all(
            len(x) == points for x in [results.radial, results.intensity, results.sigma]
        )


@pytest.mark.parametrize("radial_range", [(0, 90), (5, 38), (-10, 10)])
def test_integrate1d_radial_range(radial_range):
    arcfai = ArcFAI()
    data = np.ones((6604, 768))
    arcfai.add_imgs_to_goniometer_refinements([data], [0])
    result = arcfai.integrate1d(radial_range=radial_range, npt=1000)
    actual_x0 = max(radial_range[0], 0)
    step_size = (radial_range[1] - actual_x0) / 1000
    assert result.radial[0] == approx(actual_x0 + step_size / 2)
    assert result.radial[-1] == approx(radial_range[1] - step_size / 2)


@pytest.mark.parametrize(
    "radial_range, npt", [((0, 90), 1000), ((5, 38), 5417), ((1, 10), 1233)]
)
def test_integrate1d_step_size(radial_range, npt):
    arcfai = ArcFAI()
    data = np.ones((6604, 768))
    arcfai.add_imgs_to_goniometer_refinements([data], [0])
    result = arcfai.integrate1d(radial_range=radial_range, npt=npt)
    step_size = (radial_range[1] - radial_range[0]) / npt
    assert result.radial[0] == approx(radial_range[0] + step_size / 2)
    assert result.radial[-1] == approx(radial_range[1] - step_size / 2)
    assert result.radial[1] - result.radial[0] == approx(step_size)


def test_actual_integration(data_directory):
    """
    test adding an actual image results in something reasonable looking
    """
    arcfai = ArcFAI(from_json=data_directory / "arc_test_json_names.json")
    with h5.File(data_directory / "i15-1-53013.nxs", "r") as f:
        data = f["/entry/arc1AD/data"][0]
        tth = f["/entry/arc1AD/dummy_tth_value"][0]
    assert len(data) == len(tth) == 2
    for d, t in zip(data, tth):
        arcfai.add_imgs_to_goniometer_refinements([d], [t])
    result = arcfai.integrate1d()
    previous_result = np.loadtxt(data_directory / "example.txt").T
    assert result.radial == approx(previous_result[0])
    assert result.intensity == approx(previous_result[1])


def test_adding_separately_equals_adding_together(data_directory):
    """
    test that adding the data sequentially gives the same result as adding together
    """
    arcfai1 = ArcFAI(from_json=data_directory / "arc_test_json_names.json")
    arcfai2 = ArcFAI(from_json=data_directory / "arc_test_json_names.json")
    with h5.File(data_directory / "i15-1-53013.nxs", "r") as f:
        data = f["/entry/arc1AD/data"][0]
        tth = f["/entry/arc1AD/dummy_tth_value"][0]
    assert len(data) == len(tth) == 2
    for d, t in zip(data, tth):
        arcfai1.add_imgs_to_goniometer_refinements([d], [t])
    result1 = arcfai1.integrate1d()
    arcfai2.add_imgs_to_goniometer_refinements(data, tth)
    result2 = arcfai2.integrate1d()
    assert result1.radial == approx(result2.radial)
