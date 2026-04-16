"""Pytests for 1D recursive solution."""

from simpeg.electromagnetics import natural_source as nsem
from simpeg.electromagnetics.natural_source.sources import Planewave, PlanewaveXYPrimary
from simpeg.electromagnetics.natural_source.receivers import (
    Impedance,
    Admittance,
    Tipper,
    ApparentConductivity,
)
from simpeg import maps
import numpy as np
from scipy.constants import mu_0
import pytest


def create_survey(freq, orientation):
    """Generate test survey."""
    receivers_list = [
        Impedance([[]], component="real", orientation=orientation),
        Impedance([[]], component="imag", orientation=orientation),
        Impedance([[]], component="app_res", orientation=orientation),
        Impedance([[]], component="phase", orientation=orientation),
    ]

    source_list = [Planewave(receivers_list, f) for f in freq]

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


@pytest.mark.parametrize(
    "src_class",
    [PlanewaveXYPrimary, Planewave],
)
def test_incorrect_src_types(src_class):
    """Test incorrect source types."""
    loc = np.zeros((1, 3))
    rx = Impedance(loc)
    src = src_class(rx, frequency=10)
    survey = nsem.Survey(src)

    if src_class is not Planewave:
        with pytest.raises(
            NotImplementedError,
            match=(
                "Simulation1DRecursive defines sources using the Planewave class,"
                f" got {type(src)} instead."
            ),
        ):
            nsem.Simulation1DRecursive(survey=survey)


@pytest.mark.parametrize(
    "rx_class",
    [Impedance, Admittance, Tipper, ApparentConductivity],
)
def test_incorrect_rx_types(rx_class):
    """Test incorrect receiver types."""
    loc = np.zeros((1, 3))
    rx = rx_class(loc)
    source = Planewave(rx, frequency=10)
    survey = nsem.Survey(source)

    if rx_class is not Impedance:
        with pytest.raises(
            NotImplementedError,
            match=(
                "Simulation1DRecursive only supports the Impedance receiver class, "
                f"got {type(rx)} instead."
            ),
        ):
            nsem.Simulation1DRecursive(survey=survey)


@pytest.mark.parametrize(
    "rx_orientation",
    ["xx", "xy", "yx", "yy", "zx", "zy"],
)
def test_incorrect_rx_orientations(rx_orientation):
    """Test incorrect receiver orientations."""
    loc = np.zeros((1, 3))
    rx = Impedance(loc)
    source = Planewave(rx, frequency=10)
    survey = nsem.Survey(source)

    if (rx.orientation != "xy") and (rx.orientation != "yx"):
        with pytest.raises(
            NotImplementedError,
            match=(
                "Simulation1DRecursive only allows 'xy' or 'yx' for the orientation"
                f" property of Impedance receivers, got {rx.orientation}."
            ),
        ):
            nsem.Simulation1DRecursive(survey=survey)
