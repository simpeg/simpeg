import unittest
import numpy as np
from scipy.constants import mu_0
from simpeg import maps, tests
from simpeg.electromagnetics import natural_source as nsem

TOL = 1e-4
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0


def DerivJvecTest_1D(halfspace_value, freq=False, expMap=True):
    # Frequencies being measured
    frequencies = np.logspace(0, 4, 21)

    # Define a receiver for each data type as a list
    receivers_list = [
        nsem.receivers.PointNaturalSource(component="real"),
        nsem.receivers.PointNaturalSource(component="imag"),
        nsem.receivers.PointNaturalSource(component="app_res"),
        nsem.receivers.PointNaturalSource(component="phase"),
    ]

    # Use a list to define the planewave source at each frequency and assign receivers
    source_list = []
    for ii in range(0, len(frequencies)):
        source_list.append(nsem.sources.Planewave(receivers_list, frequencies[ii]))

    # Define the survey object
    survey = nsem.survey.Survey(source_list)

    # Layer thicknesses
    layer_thicknesses = np.array([200, 100])

    # Layer conductivities
    sigma_model = np.array([0.001, 0.01, 0.1])

    # Define a mapping for conductivities
    mapping = maps.Wires(("sigma", 3), ("thicknesses", 2))

    simulation = nsem.simulation_1d.Simulation1DRecursive(
        survey=survey,
        sigmaMap=mapping.sigma,
        thicknessesMap=mapping.thicknesses,
    )

    x0 = np.r_[sigma_model, layer_thicknesses]
    np.random.seed(1983)

    def fun(x):
        return simulation.dpred(x), lambda x: simulation.Jvec(x0, x)

    return tests.check_derivative(fun, x0, num=6, plotIt=False, eps=FLR)


def DerivJvecTest(halfspace_value, freq=False, expMap=True):
    survey, sig, sigBG, mesh = nsem.utils.test_utils.setup1DSurvey(
        halfspace_value, False, structure=True
    )
    simulation = nsem.Simulation1DPrimarySecondary(
        mesh, sigmaPrimary=sigBG, sigmaMap=maps.IdentityMap(mesh), survey=survey
    )
    print("Using {0} solver for the simulation".format(simulation.solver))
    print(
        "Derivative test of Jvec for eForm primary/secondary for 1d comp from {0} to {1} Hz\n".format(
            survey.frequencies[0], survey.frequencies[-1]
        )
    )

    x0 = sigBG
    np.random.seed(1983)
    survey = simulation.survey

    def fun(x):
        return simulation.dpred(x), lambda x: simulation.Jvec(x0, x)

    return tests.check_derivative(fun, x0, num=4, plotIt=False, eps=FLR)


class NSEM_DerivTests(unittest.TestCase):
    def test_derivJvec_Z1dr(self):
        self.assertTrue(DerivJvecTest(1e-2))

    def test_derivJvec_Z1d_e(self):
        self.assertTrue(DerivJvecTest_1D(1e-2))


if __name__ == "__main__":
    unittest.main()
