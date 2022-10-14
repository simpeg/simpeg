import unittest
from SimPEG.electromagnetics import natural_source as nsem
from SimPEG import maps
import numpy as np
from scipy.constants import mu_0


def create_survey(freq):

    receivers_list = [
        nsem.receivers.PointNaturalSource(component="real"),
        nsem.receivers.PointNaturalSource(component="imag"),
        nsem.receivers.PointNaturalSource(component="app_res"),
        nsem.receivers.PointNaturalSource(component="phase"),
    ]

    source_list = [nsem.sources.Planewave(receivers_list, f) for f in freq]

    return nsem.survey.Survey(source_list)


def true_solution(freq, sigma_half):

    # -ve sign can be removed if convention changes
    soln = np.r_[
        -np.sqrt(np.pi * freq * mu_0 / sigma_half),
        -np.sqrt(np.pi * freq * mu_0 / sigma_half),
        1 / sigma_half,
        45.0,
    ]

    return soln


def compute_simulation(freq, sigma_half):

    layer_thicknesses = np.array([100.0])
    conductivity_model = sigma_half * np.ones(2)
    model_mapping = maps.IdentityMap()

    survey = create_survey(np.array([freq]))

    simulation = nsem.simulation_1d.Simulation1DRecursive(
        survey=survey, thicknesses=layer_thicknesses, sigmaMap=model_mapping
    )

    dpred = simulation.dpred(conductivity_model)
    danal = true_solution(freq, sigma_half)

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


if __name__ == "__main__":
    unittest.main()
