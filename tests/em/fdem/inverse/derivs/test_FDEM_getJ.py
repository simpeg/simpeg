"""
Test the getJ method of the 3D FDEM simulations.
"""

import pytest
import numpy as np
import discretize
from simpeg import maps
import simpeg.electromagnetics.frequency_domain as fdem

SIMULATIONS = [
    fdem.Simulation3DElectricField,
    fdem.Simulation3DMagneticFluxDensity,
    fdem.Simulation3DCurrentDensity,
    fdem.Simulation3DMagneticField,
]

RECEIVERS = [
    fdem.receivers.PointElectricField,
    fdem.receivers.PointMagneticFluxDensity,
    fdem.receivers.PointMagneticField,
    fdem.receivers.PointCurrentDensity,
]


def get_simulation(simulation_class, receiver_class):
    """
    Build a small FDEM simulation with a magnetic dipole source.

    Parameters
    ----------
    simulation_class : type
        One of the 3D FDEM simulation classes.
    receiver_class : type
        One of the FDEM point receiver classes.

    Returns
    -------
    simulation : BaseFDEMSimulation
    """
    h = [(10.0, 4, -1.3), (10.0, 4), (10.0, 4, 1.3)]
    mesh = discretize.TensorMesh([h, h, h], origin="CCC")

    # Don't place receivers right on top of the source
    receiver_locations = np.array([[-15.0, 5.0, 5.0], [15.0, -5.0, -5.0]])
    receivers = [
        receiver_class(receiver_locations, orientation="z", component=component)
        for component in ("real", "imag")
    ]
    source = fdem.sources.MagDipole(
        receiver_list=receivers, frequency=100.0, location=np.array([0.0, 0.0, 0.0])
    )
    survey = fdem.Survey([source])
    return simulation_class(mesh=mesh, survey=survey, sigmaMap=maps.ExpMap(mesh))


@pytest.mark.parametrize("receiver_class", RECEIVERS)
@pytest.mark.parametrize("simulation_class", SIMULATIONS)
def test_getJ_vs_Jvec_Jtvec(simulation_class, receiver_class):
    """
    Test that the sensitivity matrix from getJ is consistent with Jvec and Jtvec.
    """
    simulation = get_simulation(simulation_class, receiver_class)
    rng = np.random.default_rng(seed=42)
    # Use a conductive model (1 S/m) so the real and imaginary data have similar
    # magnitudes and can be compared with a relative tolerance
    model = np.zeros(simulation.mesh.n_cells)
    v = rng.normal(size=model.size)
    w = rng.normal(size=simulation.survey.nD)

    fields = simulation.fields(model)
    jac = simulation.getJ(model, f=fields)

    np.testing.assert_allclose(jac @ v, simulation.Jvec(model, v, f=fields), rtol=1e-7)
    np.testing.assert_allclose(
        jac.T @ w, simulation.Jtvec(model, w, f=fields), rtol=1e-7
    )
