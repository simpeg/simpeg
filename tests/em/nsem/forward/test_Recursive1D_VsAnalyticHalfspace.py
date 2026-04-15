"""Pytests for 1D recursive solution."""

import warnings
from simpeg.electromagnetics import natural_source as nsem
from simpeg import maps
import numpy as np
from scipy.constants import mu_0
import pytest

ns_rx = nsem.receivers


def create_survey(freq, orientation):
    """Generate test survey."""
    receivers_list = [
        nsem.receivers.Impedance([[]], component="real", orientation=orientation),
        nsem.receivers.Impedance([[]], component="imag", orientation=orientation),
        nsem.receivers.Impedance([[]], component="app_res", orientation=orientation),
        nsem.receivers.Impedance([[]], component="phase", orientation=orientation),
    ]

    source_list = [nsem.sources.BasePlanewave(receivers_list, f) for f in freq]

    return nsem.survey.Survey(source_list)


def true_solution(freq, sigma_half, orientation):
    """Compute true solution."""
    if orientation == "yx":
        pm = -1
    else:
        pm = 1

    soln = np.r_[
        -pm * np.sqrt(np.pi * freq * mu_0 / sigma_half),
        -pm * np.sqrt(np.pi * freq * mu_0 / sigma_half),
        1 / sigma_half,
        -pm * 135.0,
    ]
    return soln


def compute_simulation(freq, sigma_half, orientation):
    """Compute numerical solution."""
    layer_thicknesses = np.array([100.0])
    conductivity_model = sigma_half * np.ones(2)
    model_mapping = maps.IdentityMap()

    survey = create_survey(np.array([freq]), orientation=orientation)

    simulation = nsem.simulation_1d.Simulation1DRecursive(
        survey=survey, thicknesses=layer_thicknesses, sigmaMap=model_mapping
    )

    dpred = simulation.dpred(conductivity_model)
    danal = true_solution(freq, sigma_half, orientation)

    return dpred, danal


# --- Forward tests (parameterized) ---
@pytest.mark.parametrize(
    "freq, sigma_half, orientation",
    [
        (0.1, 0.001, "xy"),
        (0.1, 1.0, "yx"),
        (100.0, 0.001, "xy"),
        (100.0, 1.0, "yx"),
    ],
)
def test_recursive_forward(freq, sigma_half, orientation):
    """Test recursive forward solution."""
    dpred, danal = compute_simulation(freq, sigma_half, orientation)
    np.testing.assert_allclose(dpred, danal)


# --- Receiver type validation ---
@pytest.mark.parametrize(
    "rx_class",
    [
        ns_rx.Impedance,
        ns_rx.Admittance,
        ns_rx.Tipper,
        ns_rx.ApparentConductivity,
    ],
)
def test_incorrect_rx_types(rx_class):
    """Test incorrect receiver types."""
    loc = np.zeros((1, 3))
    rx = rx_class(loc)
    source = nsem.sources.BasePlanewave(rx, frequency=10)
    survey = nsem.Survey(source)

    if rx_class is ns_rx.Impedance:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            nsem.Simulation1DRecursive(survey=survey)
    else:
        with pytest.raises(
            NotImplementedError,
            match="Simulation1DRecursive does not support .*",
        ):
            nsem.Simulation1DRecursive(survey=survey)
