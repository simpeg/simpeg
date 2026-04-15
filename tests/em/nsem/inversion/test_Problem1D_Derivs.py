"""Test derivatives for 1D simulations."""

import numpy as np
from scipy.constants import mu_0
from simpeg import maps, tests
from simpeg.electromagnetics import natural_source as nsem
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
    """Return test receivers list."""
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
    """Simulate data with recursie solution."""
    layer_thicknesses = np.array([200, 100])
    mapping = maps.Wires(("sigma", 3), ("thicknesses", 2))

    sim = nsem.simulation_1d.Simulation1DRecursive(
        survey=survey_1d,
        sigmaMap=mapping.sigma,
        thicknessesMap=mapping.thicknesses,
    )

    sigma_model = np.array([0.001, 0.01, 0.1])
    x0 = np.r_[sigma_model, layer_thicknesses]

    return sim, x0


@pytest.fixture
def primary_secondary_setup():
    """Set up primary-secondary simulation."""
    survey, sig, sigBG, mesh = nsem.utils.test_utils.setup1DSurvey(
        1e-2, False, structure=True
    )

    sim = nsem.Simulation1DPrimarySecondary(
        mesh,
        sigmaPrimary=sigBG,
        sigmaMap=maps.IdentityMap(mesh),
        survey=survey,
    )

    return sim, sigBG


# --- Tests ---
def test_derivJvec_Z1d_e(simulation_recursive):
    """Test formulation derivative."""
    simulation, x0 = simulation_recursive

    def fun(x):
        return simulation.dpred(x), lambda v: simulation.Jvec(x0, v)

    result = tests.check_derivative(
        fun, x0, num=6, plotIt=False, eps=FLR, random_seed=298376
    )

    assert result


def test_derivJvec_Z1dr(primary_secondary_setup):
    """Test other formulation derivative."""
    simulation, x0 = primary_secondary_setup

    survey = simulation.survey

    print(f"Using {simulation.solver} solver for the simulation")
    print(
        f"Derivative test of Jvec for eForm primary/secondary for 1d comp "
        f"from {survey.frequencies[0]} to {survey.frequencies[-1]} Hz\n"
    )

    def fun(x):
        return simulation.dpred(x), lambda v: simulation.Jvec(x0, v)

    result = tests.check_derivative(
        fun, x0, num=4, plotIt=False, eps=FLR, random_seed=5553
    )

    assert result
