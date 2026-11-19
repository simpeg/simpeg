import numpy as np
import pytest
from scipy.sparse import diags

from simpeg import maps
from .test_EM1D_FD_getJ import (
    create_simulation_and_conductivities as frequency_simulation,
)
from .test_EM1D_TD_getJ import create_simulation_and_conductivities as time_simulation
from .test_EM1D_TD_multilayer_ip import SURVEY_CONFIG

GROUND = SURVEY_CONFIG["ground"]


def multilayer_model(factory, eta):
    """Use the multilayer IP earth model with either domain's test survey."""
    simulation, _ = factory(True)
    simulation.thicknesses = GROUND["thicknesses"].copy()
    # The reference resistivities are DC values; the simulation uses sigma_inf.
    sigma_inf = 1.0 / (GROUND["resistivities"] * (1 - eta))
    return simulation, sigma_inf


@pytest.mark.parametrize("factory", [frequency_simulation, time_simulation])
@pytest.mark.parametrize("zero_eta", [False, True])
def test_ip_jacobian(factory, zero_eta):
    """Check joint conductivity/IP derivatives, products, and model updates."""
    eta = np.zeros(3) if zero_eta else np.full(3, GROUND["eta"][1])
    simulation, sigma = multilayer_model(factory, eta)
    wires = maps.Wires(("sigma", 3), ("eta", 3), ("tau", 3), ("c", 3))
    simulation.sigmaMap = maps.ExpMap(nP=3) * wires.sigma
    simulation.etaMap = wires.eta
    simulation.tauMap = maps.ExpMap(nP=3) * wires.tau
    simulation.cMap = wires.c
    tau = np.full(3, GROUND["tau"][1])
    c = np.full(3, GROUND["c"][1])
    model = np.r_[np.log(sigma), eta, np.log(tau), c]
    jacobian = simulation.getJ(model)
    step = 1e-4
    for i in range(model.size):
        direction = np.eye(model.size)[i]
        finite_difference = (
            simulation.dpred(model + step * direction)
            - simulation.dpred(model - step * direction)
        ) / (2 * step)
        scale = max(np.linalg.norm(jacobian[:, i]), np.linalg.norm(jacobian) * 1e-7)
        assert np.linalg.norm(jacobian[:, i] - finite_difference) < 1e-3 * scale

    rng = np.random.default_rng(42)
    vector = rng.standard_normal(model.size)
    data_vector = rng.standard_normal(simulation.survey.nD)
    np.testing.assert_allclose(
        simulation.Jvec(model, vector), jacobian @ vector, atol=1e-10
    )
    np.testing.assert_allclose(
        simulation.Jtvec(model, data_vector), jacobian.T @ data_vector, atol=1e-10
    )
    np.testing.assert_allclose(
        simulation.getJtJdiag(model), np.sum(jacobian**2, axis=0)
    )

    changed_model = model.copy()
    changed_model[3] += 0.05
    changed_jacobian = simulation.getJ(changed_model)
    assert not np.array_equal(jacobian, changed_jacobian)
    np.testing.assert_allclose(
        simulation.getJtJdiag(changed_model), np.sum(changed_jacobian**2, axis=0)
    )


@pytest.mark.parametrize("factory", [frequency_simulation, time_simulation])
def test_weighted_ip_mapping_jtj(factory):
    """Check the weighted diagonal and its cache for separate property maps."""
    eta = np.full(3, GROUND["eta"][1])
    simulation, sigma = multilayer_model(factory, eta)
    simulation.tau = np.full(3, GROUND["tau"][1])
    simulation.c = np.full(3, GROUND["c"][1])
    wires = maps.Wires(("sigma", 3), ("eta", 3))
    simulation.sigmaMap = wires.sigma
    simulation.etaMap = wires.eta
    model = np.r_[sigma, eta]
    jacobian = simulation.getJ(model)
    np.testing.assert_allclose(
        simulation.getJtJdiag(model), np.sum(jacobian**2, axis=0)
    )
    weights = diags(np.linspace(0.5, 1.5, simulation.survey.nD))
    expected = np.sum((weights @ jacobian) ** 2, axis=0)
    cached = simulation.getJtJdiag(model)
    np.testing.assert_allclose(simulation.getJtJdiag(model, W=weights), cached)
    del simulation._gtgdiag
    np.testing.assert_allclose(simulation.getJtJdiag(model, W=weights), expected)


@pytest.mark.parametrize("factory", [frequency_simulation, time_simulation])
@pytest.mark.parametrize("name", ["eta", "tau", "c"])
def test_ip_only_mapping(factory, name):
    """Check each IP parameter's derivative with conductivity held fixed."""
    eta = np.full(3, GROUND["eta"][1])
    original, sigma = multilayer_model(factory, eta)
    values = {name: GROUND[name][1] for name in ("eta", "tau", "c")}
    model = np.full(3, values.pop(name))
    simulation = type(original)(
        survey=original.survey,
        thicknesses=original.thicknesses,
        sigma=sigma,
        **values,
        **{name + "Map": maps.IdentityMap(nP=3)},
    )
    jacobian = simulation.getJ(model)
    direction = np.array([0.3, -0.2, 0.4])
    step = 1e-6 if name == "tau" else 1e-4
    finite_difference = (
        simulation.dpred(model + step * direction)
        - simulation.dpred(model - step * direction)
    ) / (2 * step)
    expected = jacobian @ direction
    assert np.linalg.norm(finite_difference - expected) < 1e-3 * np.linalg.norm(
        expected
    )


@pytest.mark.parametrize("factory", [frequency_simulation, time_simulation])
def test_heterogeneous_multilayer_ip_jacobian(factory):
    """Check derivatives for the reference model's chargeable middle layer."""
    eta = GROUND["eta"].copy()
    simulation, sigma = multilayer_model(factory, eta)
    wires = maps.Wires(("sigma", 3), ("eta", 1), ("tau", 1), ("c", 1))
    inject = maps.InjectActiveCells(None, active_cells=[1], nC=3)
    simulation.sigmaMap = maps.ExpMap(nP=3) * wires.sigma
    simulation.etaMap = inject * wires.eta
    simulation.tauMap = inject * maps.ExpMap(nP=1) * wires.tau
    simulation.cMap = inject * wires.c
    model = np.r_[np.log(sigma), eta[1], np.log(GROUND["tau"][1]), GROUND["c"][1]]
    jacobian = simulation.getJ(model)
    np.testing.assert_allclose(simulation.eta, GROUND["eta"])
    np.testing.assert_allclose(simulation.tau, GROUND["tau"])
    np.testing.assert_allclose(simulation.c, GROUND["c"])
    step = 1e-4
    for i in range(model.size):
        direction = np.eye(model.size)[i]
        finite_difference = (
            simulation.dpred(model + step * direction)
            - simulation.dpred(model - step * direction)
        ) / (2 * step)
        scale = max(np.linalg.norm(jacobian[:, i]), np.linalg.norm(jacobian) * 1e-7)
        assert np.linalg.norm(jacobian[:, i] - finite_difference) < 1e-3 * scale
