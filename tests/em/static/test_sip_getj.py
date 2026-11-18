"""Regression tests for the spatial SIP sensitivity matrix."""

import discretize
import numpy as np
import pytest
from numpy.testing import assert_allclose
from pymatsolver import SolverLU

from simpeg import maps
from simpeg.electromagnetics import spectral_induced_polarization as sip
from simpeg.electromagnetics.static.spectral_induced_polarization.data import Data


@pytest.mark.parametrize(
    "simulation_class", [sip.Simulation3DNodal, sip.Simulation3DCellCentered]
)
@pytest.mark.parametrize("active_cells", [None, "all", "subset"])
@pytest.mark.parametrize("store_j", [False, True])
def test_getj(simulation_class, active_cells, store_j):
    """Compare explicit sensitivities with the matrix-free adjoint operation."""
    mesh = discretize.TensorMesh([[(25.0, 4)]] * 3, x0="CCN")
    times = np.array([0.01, 0.02])
    receivers = [
        sip.receivers.Pole([[0.0, 0.0, 0.0], [25.0, 0.0, 0.0]], times),
        sip.receivers.Dipole([[-12.5, 0.0, 0.0]], [[12.5, 0.0, 0.0]], times),
    ]
    source = sip.sources.Dipole(receivers, [-37.5, 0.0, 0.0], [37.5, 0.0, 0.0])
    survey = sip.Survey([source])
    active = None
    if active_cells is not None:
        active = np.ones(mesh.n_cells, dtype=bool)
        if active_cells == "subset":
            active = mesh.cell_centers[:, 2] < -50.0

    wires = maps.Wires(("eta", mesh.n_cells), ("taui", mesh.n_cells))
    kwargs = dict(
        mesh=mesh,
        survey=survey,
        sigma=np.full(mesh.n_cells, 0.01),
        etaMap=wires.eta,
        tauiMap=wires.taui,
        solver=SolverLU,
    )
    reference = simulation_class(**kwargs)
    simulation = simulation_class(**kwargs, storeJ=store_j, actinds=active)
    eta = np.linspace(0.01, 0.1, mesh.n_cells)
    if store_j and active is not None:
        eta[~active] = 0.0
    model = np.r_[eta, np.full(mesh.n_cells, 100.0)]

    fields = reference.fields(model)
    predicted = reference.dpred(model, f=fields)
    n_locations = survey.n_locations
    expected = np.empty((n_locations, mesh.n_cells))
    decay = reference.get_peta_eta_deriv(times[0])
    i = 0
    for receiver in receivers:
        for j in range(receiver.nD):
            data_vector = Data(survey, np.zeros(survey.nD))
            values = np.zeros(receiver.nD)
            values[j] = 1.0
            data_vector[source, receiver, times[0]] = values
            expected[i] = (
                reference.Jtvec(model, data_vector, f=fields)[: mesh.n_cells] / decay
            )
            i += 1
    if store_j and active is not None:
        expected = expected[:, active]

    jacobian = simulation.getJ(model)
    assert jacobian.shape == expected.shape
    assert np.isfinite(jacobian).all()
    assert_allclose(jacobian, expected, rtol=1e-12, atol=1e-12)

    pseudo_chargeability = reference.get_peta(times[0])
    if store_j and active is not None:
        pseudo_chargeability = pseudo_chargeability[active]
    assert_allclose(
        jacobian @ pseudo_chargeability,
        predicted[:n_locations],
        rtol=1e-12,
        atol=1e-12,
    )
    if store_j:
        assert_allclose(simulation.dpred(model), predicted, rtol=1e-12, atol=1e-12)
        direction = np.r_[eta, np.full(mesh.n_cells, 1.0)]
        assert_allclose(
            simulation.Jvec(model, direction),
            reference.Jvec(model, direction, f=fields),
            rtol=1e-12,
            atol=1e-12,
        )
