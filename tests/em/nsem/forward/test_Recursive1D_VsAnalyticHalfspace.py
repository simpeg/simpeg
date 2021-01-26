import unittest
from SimPEG.electromagnetics import natural_source as nsem
from SimPEG import maps
import numpy as np
from scipy.constants import mu_0


def create_survey(freq):

    receivers_list = [
        nsem.receivers.AnalyticReceiver1D(component='real'),
        nsem.receivers.AnalyticReceiver1D(component='imag'),
        nsem.receivers.AnalyticReceiver1D(component='app_res'),
        nsem.receivers.AnalyticReceiver1D(component='phase')
        ]

    source_list = [nsem.sources.AnalyticPlanewave1D(receivers_list, freq)]

    return nsem.survey.Survey1D(source_list)

def true_solution(freq, sigma_half):

    # -ve sign can be removed if convention changes
    soln = np.r_[
        -np.sqrt(np.pi*freq*mu_0/sigma_half),
        -np.sqrt(np.pi*freq*mu_0/sigma_half),
        1/sigma_half,
        45.
        ]

    return soln


def compute_simulation_error(freq, sigma_half):

    layer_thicknesses = np.array([100.])
    conductivity_model = sigma_half*np.ones(2)
    model_mapping = maps.IdentityMap()

    survey = create_survey(np.array([freq]))

    simulation = nsem.simulation_1d.Simulation1DRecursive(
        survey=survey, thicknesses=layer_thicknesses, sigmaMap=model_mapping
    )

    dpred = simulation.dpred(conductivity_model)
    danal = true_solution(freq, sigma_half)

    return np.abs((danal - dpred)/danal)


class TestRecursiveForward(unittest.TestCase):

    def test_1(self):
        self.assertTrue(np.all(compute_simulation_error(0.1, 0.001) < 1e-4))

    def test_2(self):
        self.assertTrue(np.all(compute_simulation_error(0.1, 1.) < 1e-4))

    def test_3(self):
        self.assertTrue(np.all(compute_simulation_error(100., 0.001) < 1e-4))

    def test_4(self):
        self.assertTrue(np.all(compute_simulation_error(100., 1.) < 1e-4))


if __name__ == '__main__':
    unittest.main()
