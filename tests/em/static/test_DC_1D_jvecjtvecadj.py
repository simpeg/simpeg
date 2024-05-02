from discretize.tests import check_derivative, assert_isadjoint
import numpy as np
import pytest
from simpeg import (
    maps,
)
from simpeg.electromagnetics import resistivity as dc

TOL = 1e-5
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order


def get_survey(data_type, rx_type, tx_type):
    n_spacings = np.logspace(0, 2, 3)
    y = np.zeros_like(n_spacings)
    z = np.full_like(n_spacings, -1.0)
    a_locations = np.column_stack((-1.5 * n_spacings, y, z))
    b_locations = np.column_stack((1.5 * n_spacings, y, z))
    m_locations = np.column_stack((-0.5 * n_spacings, y, z))
    n_locations = np.column_stack((0.5 * n_spacings, y, z))

    # some dipole receivers (with and without app_res))
    sources = []
    if data_type == "both":
        data_type = ["volt", "apparent_resistivity"]
    else:
        data_type = [data_type]
    if rx_type == "both":
        rx_type = ["p", "d"]
    else:
        rx_type = [rx_type]
    if tx_type == "both":
        tx_type = ["p", "d"]
    else:
        tx_type = [tx_type]
    for a, b, m, n in zip(a_locations, b_locations, m_locations, n_locations):
        rx = []
        for dtype in data_type:
            for rxt in rx_type:
                if rxt == "p":
                    rx.append(dc.receivers.Pole(m, data_type=dtype))
                else:
                    rx.append(dc.receivers.Dipole(locations=[m, n], data_type=dtype))

        for txt in tx_type:
            if txt == "p":
                sources.append(dc.sources.Pole(rx, a))
            else:
                sources.append(dc.sources.Dipole(rx, location=[a, b]))
    return dc.Survey(sources)


@pytest.mark.parametrize("rx_type", ["p", "d", "both"])
@pytest.mark.parametrize("tx_type", ["p", "d", "both"])
@pytest.mark.parametrize("data_type", ["volt", "apparent_resistivity", "both"])
def test_forward_accuracy(data_type, rx_type, tx_type):
    sigma = 1.0
    cond = [sigma, sigma, sigma]
    thick = [10, 20]
    survey = get_survey(data_type, rx_type, tx_type)
    simulation = dc.Simulation1DLayers(
        survey=survey,
        sigma=cond,
        thicknesses=thick,
    )
    d = simulation.dpred()

    # apparent resistivity should be close to the true resistivity of this half space
    if data_type == "app_res":
        np.testing.assert_allclose(d, 1.0 / sigma, rtol=1e-6)
    if data_type == "volt":
        geom = dc.utils.geometric_factor(survey)
        np.testing.assert_allclose(d / geom, 1.0 / sigma, rtol=1e-6)
    if data_type == "both":
        n_r = 2 if rx_type == "both" else 1
        d = d.reshape(-1, n_r)
        geom = dc.utils.geometric_factor(survey).reshape(-1, n_r)[::2]
        # double check the app_res calc was applied correctly
        d_volts = d[::2]
        d_app = d[1::2]
        np.testing.assert_allclose(d_volts / geom, d_app)

        # double check the accuracy
        np.testing.assert_allclose(d_app, 1.0 / sigma, rtol=1e-6)


@pytest.mark.parametrize("deriv_type", ("sigma", "h", "both"))
def test_derivative(deriv_type):
    n_layer = 4
    np.random.seed(40)
    log_cond = np.random.rand(n_layer)
    log_thick = np.random.rand(n_layer - 1)

    if deriv_type != "h":
        sigma_map = maps.ExpMap()
        model = log_cond
        sigma = None
    else:
        sigma = np.exp(log_cond)
        sigma_map = None
    if deriv_type != "sigma":
        h_map = maps.ExpMap()
        model = log_thick
        h = None
    else:
        h_map = None
        h = np.exp(log_thick)
    if deriv_type == "both":
        wire = maps.Wires(("sigma", n_layer), ("thick", n_layer - 1))
        sigma_map = sigma_map * wire.sigma
        h_map = h_map * wire.thick
        model = np.r_[log_cond, log_thick]
    # already tested that the forward operation works for all combinations of src/rx/data_type
    # the way this simulation is set up, only need to test the derivative with
    # a single combination.
    survey = get_survey("volt", "d", "d")
    simulation = dc.Simulation1DLayers(
        survey=survey,
        sigma=sigma,
        sigmaMap=sigma_map,
        thicknesses=h,
        thicknessesMap=h_map,
    )

    def sim_1d_func(m):
        d = simulation.dpred(m)

        def J(v):
            return simulation.Jvec(m, v)

        return d, J

    assert check_derivative(sim_1d_func, model, plotIt=False, num=4)


@pytest.mark.parametrize("deriv_type", ("sigma", "h", "both"))
def test_adjoint(deriv_type):
    n_layer = 4
    np.random.seed(40)
    log_cond = np.random.rand(n_layer)
    log_thick = np.random.rand(n_layer - 1)

    if deriv_type != "h":
        sigma_map = maps.ExpMap()
        model = log_cond
        sigma = None
    else:
        sigma = np.exp(log_cond)
        sigma_map = None
    if deriv_type != "sigma":
        h_map = maps.ExpMap()
        model = log_thick
        h = None
    else:
        h_map = None
        h = np.exp(log_thick)
    if deriv_type == "both":
        wire = maps.Wires(("sigma", n_layer), ("thick", n_layer - 1))
        sigma_map = sigma_map * wire.sigma
        h_map = h_map * wire.thick
        model = np.r_[log_cond, log_thick]
    # already tested that the forward operation works for all combinations of src/rx/data_type
    # the way this simulation is set up, only need to test the derivative with
    # a single combination.
    survey = get_survey("volt", "d", "d")
    simulation = dc.Simulation1DLayers(
        survey=survey,
        sigma=sigma,
        sigmaMap=sigma_map,
        thicknesses=h,
        thicknessesMap=h_map,
    )

    def J(v):
        return simulation.Jvec(model, v)

    def JT(v):
        return simulation.Jtvec(model, v)

    assert_isadjoint(J, JT, len(model), survey.nD)


def test_errors():
    sigma = 1.0
    cond = [sigma, sigma, sigma]
    thick = [10, 20]
    survey = get_survey("apparent_resistivity", "d", "d")

    with pytest.raises(TypeError):
        simulation = dc.Simulation1DLayers(
            survey=survey,
            sigma=cond,
            thicknesses=thick,
            storeJ=False,
        )
    with pytest.raises(TypeError):
        simulation = dc.Simulation1DLayers(
            survey=survey,
            sigma=cond,
            thicknesses=thick,
            hankel_pts_per_dec=2,
        )
    with pytest.raises(TypeError):
        simulation = dc.Simulation1DLayers(
            survey=survey,
            sigma=cond,
            thicknesses=thick,
            data_type="volt",
        )

    simulation = dc.Simulation1DLayers(
        sigma=cond,
        thicknesses=thick,
    )
    with pytest.raises(AttributeError):
        _ = simulation.survey
    simulation.survey = survey


def test_functionality():
    n_layer = 4
    np.random.seed(40)
    log_cond = np.random.rand(n_layer)
    thick = np.random.rand(n_layer - 1)

    sigma_map = maps.ExpMap()
    model = log_cond

    survey = get_survey("volt", "d", "d")
    simulation = dc.Simulation1DLayers(
        survey=survey,
        sigmaMap=sigma_map,
        thicknesses=thick,
    )

    # test dpred with and without fields passed
    f = simulation.fields(model)
    d1 = simulation.dpred(model, f)
    d2 = simulation.dpred()
    np.testing.assert_allclose(d1, d2)

    # test clearing J on model update if fix_J is False
    simulation.fix_Jmatrix = False
    J1 = simulation.getJ(model)
    J2 = simulation.getJ(model + 2)
    assert J1 is not J2

    simulation.fix_Jmatrix = True
    J1 = simulation.getJ(model)
    J2 = simulation.getJ(model + 2)
    assert J1 is J2
