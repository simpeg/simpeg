import unittest
import warnings

from simpeg.electromagnetics import natural_source as nsem
from simpeg import maps
import numpy as np
from scipy.constants import mu_0

ns_rx = nsem.receivers

import pytest


def create_survey(freq):
    receivers_list = [
        nsem.receivers.PointNaturalSource(component="real"),
        nsem.receivers.PointNaturalSource(component="imag"),
        nsem.receivers.PointNaturalSource(component="app_res"),
        nsem.receivers.PointNaturalSource(component="phase"),
    ]

    source_list = [nsem.sources.Planewave(receivers_list, f) for f in freq]

    return nsem.survey.Survey(source_list)


def true_solution(freq, conductivity_half):
    # -ve sign can be removed if convention changes
    soln = np.r_[
        -np.sqrt(np.pi * freq * mu_0 / conductivity_half),
        -np.sqrt(np.pi * freq * mu_0 / conductivity_half),
        1 / conductivity_half,
        45.0,
    ]

    return soln


def compute_simulation(freq, conductivity_half):
    layer_thicknesses = np.array([100.0])
    conductivity_model = conductivity_half * np.ones(2)
    model_mapping = maps.IdentityMap()

    survey = create_survey(np.array([freq]))

    simulation = nsem.simulation_1d.Simulation1DRecursive(
        survey=survey, thicknesses=layer_thicknesses, conductivity_map=model_mapping
    )

    dpred = simulation.dpred(conductivity_model)
    danal = true_solution(freq, conductivity_half)

    return dpred, danal


class TestRecursiveForward(unittest.TestCase):
    def test_1(self):
        np.testing.assert_allclose(*compute_simulation(0.1, 0.001))

    def test_2(self):
        np.testing.assert_allclose(*compute_simulation(0.1, 1.0))

    def test_3(self):
        np.testing.assert_allclose(*compute_simulation(100.0, 0.001))

    def test_4(self):
        np.testing.assert_allclose(*compute_simulation(100.0, 1.0))


@pytest.mark.parametrize(
    "rx_class",
    [
        ns_rx.Impedance,
        ns_rx.PointNaturalSource,
        ns_rx.Admittance,
        ns_rx.Tipper,
        ns_rx.Point3DTipper,
        ns_rx.ApparentConductivity,
    ],
)
def test_incorrect_rx_types(rx_class):
    loc = np.zeros((1, 3))
    rx = rx_class(loc)
    source = nsem.sources.Planewave(rx, frequency=10)
    survey = nsem.Survey(source)
    # make sure that only these exact classes do not issue warnings.
    if rx_class in [ns_rx.Impedance, ns_rx.PointNaturalSource]:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            nsem.Simulation1DRecursive(survey=survey)
    else:
        with pytest.raises(
            NotImplementedError, match="Simulation1DRecursive does not support .*"
        ):
            nsem.Simulation1DRecursive(survey=survey)


if __name__ == "__main__":
    unittest.main()
