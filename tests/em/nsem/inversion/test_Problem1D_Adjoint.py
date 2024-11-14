import numpy as np
import unittest
from scipy.constants import mu_0

from simpeg.electromagnetics import natural_source as nsem
from simpeg import maps


TOL = 1e-4
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0


def JvecAdjointTest_1D(conductivityHalf, formulation="PrimSec"):
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
    conductivity_model = np.array([0.001, 0.01, 0.1])

    # Define a mapping for conductivities, thicknesses
    mapping = maps.Wires(("conductivity", 3), ("thicknesses", 2))

    simulation = nsem.simulation_1d.Simulation1DRecursive(
        survey=survey,
        conductivity_map=mapping.conductivity,
        thicknessesMap=mapping.thicknesses,
    )

    m = np.r_[conductivity_model, layer_thicknesses]
    u = simulation.fields(m)

    rng = np.random.default_rng(seed=1983)
    v = rng.uniform(size=survey.nD)
    w = rng.uniform(size=len(m))

    vJw = v.dot(simulation.Jvec(m, w, u))
    wJtv = w.dot(simulation.Jtvec(m, v, u))
    tol = np.max([TOL * (10 ** int(np.log10(np.abs(vJw)))), FLR])
    print(" vJw   wJtv  vJw - wJtv     tol    abs(vJw - wJtv) < tol")
    print(vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol)
    return np.abs(vJw - wJtv) < tol


def JvecAdjointTest(conductivityHalf, formulation="PrimSec"):
    forType = "PrimSec" not in formulation
    survey, conductivity, sigBG, m1d = nsem.utils.test_utils.setup1DSurvey(
        conductivityHalf, tD=forType, structure=False
    )
    print("Adjoint test of e formulation for {:s} comp \n".format(formulation))

    if "PrimSec" in formulation:
        problem = nsem.Simulation1DPrimarySecondary(
            m1d,
            survey=survey,
            conductivityPrimary=sigBG,
            conductivity_map=maps.IdentityMap(m1d),
        )
    else:
        raise NotImplementedError(
            "Only {} formulations are implemented.".format(formulation)
        )
    m = conductivity
    u = problem.fields(m)

    rng = np.random.default_rng(seed=1983)
    v = rng.uniform(size=survey.nD)
    # print problem.PropMap.PropModel.nP
    w = rng.uniform(size=problem.mesh.nC)

    vJw = v.ravel().dot(problem.Jvec(m, w, u))
    wJtv = w.ravel().dot(problem.Jtvec(m, v, u))
    tol = np.max([TOL * (10 ** int(np.log10(np.abs(vJw)))), FLR])
    print(" vJw   wJtv  vJw - wJtv     tol    abs(vJw - wJtv) < tol")
    print(vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol)
    return np.abs(vJw - wJtv) < tol


class NSEM_1D_AdjointTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_JvecAdjoint_All(self):
        self.assertTrue(JvecAdjointTest(1e-2))

    def test_JvecAdjoint_All_1D(self):
        self.assertTrue(JvecAdjointTest_1D(1e-2))


if __name__ == "__main__":
    unittest.main()
