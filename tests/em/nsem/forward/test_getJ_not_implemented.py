"""
Test NotImplementedError on getJ for NSEM 1D finite volume simulations.
"""

import pytest
import numpy as np
import discretize
from simpeg import maps
from simpeg.electromagnetics import natural_source as nsem


@pytest.fixture
def mesh():
    csz = 100
    nc = 300
    npad = 30
    pf = 1.2
    mesh = discretize.TensorMesh([[(csz, npad, -pf), (csz, nc), (csz, npad)]], "N")
    mesh.x0 = np.r_[-mesh.h[0][:-npad].sum()]
    return mesh


@pytest.fixture
def survey():
    frequencies = np.logspace(-2, 1, 30)
    receiver = nsem.receivers.Impedance(
        [[0]], orientation="xy", component="apparent_resistivity"
    )
    sources = [nsem.sources.Planewave([receiver], frequency=f) for f in frequencies]
    survey = nsem.survey.Survey(sources)
    return survey


@pytest.mark.parametrize(
    "simulation_class", [nsem.Simulation1DElectricField, nsem.Simulation1DMagneticField]
)
def test_getJ_not_implemented(mesh, survey, simulation_class):
    """
    Test NotImplementedError on getJ for NSEM 1D simulations.
    """
    mapping = maps.IdentityMap()
    simulation = simulation_class(
        mesh=mesh,
        survey=survey,
        sigmaMap=mapping,
    )
    model = np.ones(survey.nD)
    msg = "The getJ method hasn't been implemented"
    with pytest.raises(NotImplementedError, match=msg):
        simulation.getJ(model)
