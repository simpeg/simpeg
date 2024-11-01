import inspect
import re

import pytest
import simpeg.electromagnetics.natural_source as nsem
import numpy as np
import discretize
import numpy.testing as npt


@pytest.fixture(
    params=[
        "same_location",
        "diff_location",
    ]
)
def impedance_pairs(request):
    test_e_locs = np.array([[0.2, 0.1, 0.3], [-0.1, 0.2, -0.3]])
    test_h_locs = np.array([[-0.2, 0.24, 0.1], [0.5, 0.2, -0.2]])

    rx_point_type = request.param
    if rx_point_type == "same":
        rx1 = nsem.receivers.PointNaturalSource(test_e_locs)
        rx2 = nsem.receivers.Impedance(test_e_locs, orientation="xy")
    else:
        rx1 = nsem.receivers.PointNaturalSource(
            locations_e=test_e_locs, locations_h=test_h_locs
        )
        rx2 = nsem.receivers.Impedance(
            locations_e=test_e_locs, locations_h=test_h_locs, orientation="xy"
        )
    return rx1, rx2


@pytest.fixture()
def tipper_pairs():
    test_e_locs = np.array([[0.2, 0.1, 0.3], [-0.1, 0.2, -0.3]])

    rx1 = nsem.receivers.Point3DTipper(test_e_locs)
    rx2 = nsem.receivers.Tipper(test_e_locs, orientation="zx")
    return rx1, rx2


def test_deprecation():
    test_loc = np.array([10.0, 11.0, 12.0])
    with pytest.warns(FutureWarning, match="PointNaturalSource has been deprecated.*"):
        nsem.receivers.PointNaturalSource(test_loc)

    with pytest.warns(FutureWarning, match="Using the default for locations.*"):
        nsem.receivers.PointNaturalSource()

    with pytest.warns(FutureWarning, match="Point3DTipper has been deprecated.*"):
        nsem.receivers.Point3DTipper(test_loc)


def test_imp_consistent_attributes(impedance_pairs):
    rx1, rx2 = impedance_pairs

    for item_name in dir(rx1):
        is_dunder = re.match(r"__\w+__", item_name) is not None
        # skip a few things related to the wrapping, and dunder methods
        if not (item_name in ["locations", "_uid", "uid", "_old__init__"] or is_dunder):
            item1 = getattr(rx1, item_name)
            item2 = getattr(rx2, item_name)
            if not (inspect.isfunction(item1) or inspect.ismethod(item1)):
                if isinstance(item1, np.ndarray):
                    npt.assert_array_equal(item1, item2)
                else:
                    assert item1 == item2

    npt.assert_array_equal(rx1.locations, rx2.locations_e)


def test_tip_consistent_attributes(tipper_pairs):
    rx1, rx2 = tipper_pairs

    for item_name in dir(rx1):
        is_dunder = re.match(r"__\w+__", item_name) is not None
        # skip a few things related to the wrapping, and dunder methods
        if not (
            item_name in ["locations", "locations_e", "_uid", "uid", "_old__init__"]
            or is_dunder
        ):
            item1 = getattr(rx1, item_name)
            item2 = getattr(rx2, item_name)
            if not (inspect.isfunction(item1) or inspect.ismethod(item1)):
                print(item_name, item1, item2)
                if isinstance(item1, np.ndarray):
                    npt.assert_array_equal(item1, item2)
                else:
                    assert item1 == item2

    npt.assert_array_equal(rx1.locations, rx2.locations_h)
    npt.assert_array_equal(rx1.locations, rx2.locations_base)


@pytest.mark.parametrize(
    "rx_component", ["real", "imag", "apparent_resistivity", "phase", "complex"]
)
def test_imp_consistent_eval(impedance_pairs, rx_component):
    rx1, rx2 = impedance_pairs
    rx1.component = rx_component
    rx2.component = rx_component
    # test that the output of the function eval returns the same thing,
    # since it was updated...
    mesh = discretize.TensorMesh([3, 4, 5], origin="CCC")

    # create a mock simulation
    src = nsem.sources.PlanewaveXYPrimary(
        [rx1, rx2], frequency=10, sigma_primary=np.ones(mesh.n_cells)
    )
    survey = nsem.Survey(src)
    sim_temp = nsem.Simulation3DPrimarySecondary(survey=survey, mesh=mesh, sigma=1)

    # Create a mock field,
    f = sim_temp.fieldsPair(sim_temp)
    test_u = np.linspace(1, 2, 2 * mesh.n_edges) + 1j * np.linspace(
        -1, 1, 2 * mesh.n_edges
    )
    f[src, sim_temp._solutionType] = test_u.reshape(mesh.n_edges, 2)

    v1 = rx1.eval(src, mesh, f)
    v2 = rx2.eval(src, mesh, f)

    npt.assert_equal(v1, v2)

    if rx_component == "real":
        # do a quick test here that calling eval on rx1 is the same as calling
        # eval on rx2 with a complex component
        rx2.component = "complex"
        with pytest.warns(FutureWarning, match="Calling with return_complex=True.*"):
            v1 = rx1.eval(src, mesh, f, return_complex=True)
        v2 = rx2.eval(src, mesh, f)

        # assert it reset
        assert rx1.component == "real"
        # assert the outputs are the same
        npt.assert_equal(v1, v2)


@pytest.mark.parametrize("rx_component", ["real", "imag", "complex"])
def test_tip_consistent_eval(tipper_pairs, rx_component):
    rx1, rx2 = tipper_pairs
    rx1.component = rx_component
    rx2.component = rx_component
    # test that the output of the function eval returns the same thing,
    # since it was updated...
    mesh = discretize.TensorMesh([3, 4, 5], origin="CCC")

    # create a mock simulation
    src = nsem.sources.PlanewaveXYPrimary(
        [rx1, rx2], frequency=10, sigma_primary=np.ones(mesh.n_cells)
    )
    survey = nsem.Survey(src)
    sim_temp = nsem.Simulation3DPrimarySecondary(survey=survey, mesh=mesh, sigma=1)

    # Create a mock field,
    f = sim_temp.fieldsPair(sim_temp)
    test_u = np.linspace(1, 2, 2 * mesh.n_edges) + 1j * np.linspace(
        -1, 1, 2 * mesh.n_edges
    )
    f[src, sim_temp._solutionType] = test_u.reshape(mesh.n_edges, 2)

    v1 = rx1.eval(src, mesh, f)
    v2 = rx2.eval(src, mesh, f)

    npt.assert_equal(v1, v2)

    if rx_component == "real":
        # do a quick test here that calling eval on rx1 is the same as calling
        # eval on rx2 with a complex component
        rx2.component = "complex"
        with pytest.warns(FutureWarning, match="Calling with return_complex=True.*"):
            v1 = rx1.eval(src, mesh, f, return_complex=True)
        v2 = rx2.eval(src, mesh, f)

        # assert it reset
        assert rx1.component == "real"
        # assert the outputs are the same
        npt.assert_equal(v1, v2)


def test_imp_location_initialization():
    loc_1 = np.empty((2, 3))
    loc_2 = np.empty((2, 3))
    with pytest.raises(TypeError, match="Cannot pass both locations and .*"):
        nsem.receivers.PointNaturalSource(locations=loc_1, locations_h=loc_2)

    with pytest.raises(TypeError, match="Either locations or both locations_e.*"):
        nsem.receivers.PointNaturalSource(locations_e=loc_1)

    rx1 = nsem.receivers.PointNaturalSource(locations=[loc_1])
    rx2 = nsem.receivers.Impedance(loc_1)
    npt.assert_equal(rx1.locations, rx2.locations_e)
    npt.assert_equal(rx1.locations, rx2.locations_h)

    rx1 = nsem.receivers.PointNaturalSource(locations=[loc_1, loc_2])
    rx2 = nsem.receivers.Impedance(loc_1, loc_2)
    npt.assert_equal(rx1.locations_e, rx2.locations_e)
    npt.assert_equal(rx1.locations_h, rx2.locations_h)


def test_tip_location_initialization():
    loc_1 = np.empty((2, 3))
    loc_2 = np.empty((2, 3))
    with pytest.warns(UserWarning, match="locations_e and locations_h are unused.*"):
        nsem.receivers.Point3DTipper(locations=loc_1, locations_e=loc_2)

    with pytest.raises(
        ValueError, match="incorrect size of list, must be length of 1 or 2"
    ):
        nsem.receivers.Point3DTipper([loc_1, loc_1, loc_1])
