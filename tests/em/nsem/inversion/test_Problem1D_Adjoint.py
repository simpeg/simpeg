"""Adjoint tests for 1D simulations."""

import numpy as np
from scipy.constants import mu_0
from simpeg.electromagnetics import natural_source as nsem
from simpeg import maps
import pytest

TOL = 1e-4
FLR = 1e-20
CONDUCTIVITY = 1e1
MU = mu_0


# --- Fixtures ---
@pytest.fixture
def frequencies():
    """Return test frequencies."""
    return np.logspace(0, 4, 21)


@pytest.fixture
def receivers_list():
    """Generate receivers list."""
    return [
        nsem.receivers.Impedance([[]], component="real", orientation="xy"),
        nsem.receivers.Impedance([[]], component="imag", orientation="xy"),
        nsem.receivers.Impedance([[]], component="app_res", orientation="xy"),
        nsem.receivers.Impedance([[]], component="phase", orientation="xy"),
        nsem.receivers.Impedance([[]], component="real", orientation="yx"),
        nsem.receivers.Impedance([[]], component="imag", orientation="yx"),
        nsem.receivers.Impedance([[]], component="app_res", orientation="yx"),
        nsem.receivers.Impedance([[]], component="phase", orientation="yx"),
    ]


@pytest.fixture
def survey_1d(frequencies, receivers_list):
    """Generate 1D survey."""
    source_list = [nsem.sources.BasePlanewave(receivers_list, f) for f in frequencies]
    return nsem.survey.Survey(source_list)


@pytest.fixture
def simulation_recursive(survey_1d):
    """Set up recursive 1D simulation."""
    layer_thicknesses = np.array([200, 100])
    sigma_model = np.array([0.001, 0.01, 0.1])

    mapping = maps.Wires(("sigma", 3), ("thicknesses", 2))

    sim = nsem.simulation_1d.Simulation1DRecursive(
        survey=survey_1d,
        sigmaMap=mapping.sigma,
        thicknessesMap=mapping.thicknesses,
    )

    m = np.r_[sigma_model, layer_thicknesses]
    u = sim.fields(m)

    return sim, m, u


@pytest.fixture
def primary_secondary_setup():
    """Set up primary-secondary simulation."""
    survey, sigma, sigBG, mesh = nsem.utils.test_utils.setup1DSurvey(
        1e-2, tD=False, structure=False
    )

    sim = nsem.Simulation1DPrimarySecondary(
        mesh,
        survey=survey,
        sigmaPrimary=sigBG,
        sigmaMap=maps.IdentityMap(mesh),
    )

    m = sigma
    u = sim.fields(m)

    return sim, m, u, survey


# --- Tests ---
def test_JvecAdjoint_All(primary_secondary_setup):
    """Test adjoint all."""
    simulation, m, u, survey = primary_secondary_setup

    rng = np.random.default_rng(seed=1983)
    v = rng.uniform(size=survey.nD)
    w = rng.uniform(size=simulation.mesh.nC)

    vJw = v.ravel().dot(simulation.Jvec(m, w, u))
    wJtv = w.ravel().dot(simulation.Jtvec(m, v, u))

    tol = np.max([TOL * (10 ** int(np.log10(np.abs(vJw)))), FLR])

    print(" vJw   wJtv  vJw - wJtv     tol    abs(vJw - wJtv) < tol")
    print(vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol)

    np.testing.assert_allclose(vJw, wJtv, atol=TOL)


def test_JvecAdjoint_All_1D(simulation_recursive):
    """Test adjoint for 1D recursie."""
    simulation, m, u = simulation_recursive

    rng = np.random.default_rng(seed=1983)
    v = rng.uniform(size=simulation.survey.nD)
    w = rng.uniform(size=len(m))

    vJw = v.dot(simulation.Jvec(m, w, u))
    wJtv = w.dot(simulation.Jtvec(m, v, u))

    tol = np.max([TOL * (10 ** int(np.log10(np.abs(vJw)))), FLR])

    print(" vJw   wJtv  vJw - wJtv     tol    abs(vJw - wJtv) < tol")
    print(vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol)

    np.testing.assert_allclose(vJw, wJtv, atol=TOL)
