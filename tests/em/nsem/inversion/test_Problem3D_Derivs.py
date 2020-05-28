from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# Test functions
import unittest
import numpy as np
from SimPEG import tests, mkvc
from SimPEG.electromagnetics import natural_source as nsem
from scipy.constants import mu_0

TOLr = 5e-2
TOL = 1e-4
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0

# Test the Jvec derivative
def DerivJvecTest(inputSetup, comp="All", freq=False, expMap=True):
    (M, freqs, sig, sigBG, rx_loc) = inputSetup
    survey, simulation = nsem.utils.test_utils.setupSimpegNSEM_ePrimSec(
        inputSetup, comp=comp, singleFreq=freq, expMap=expMap
    )
    print("Using {0} solver for the simulation".format(simulation.Solver))
    print(
        "Derivative test of Jvec for eForm primary/secondary for {} comp at {}\n".format(
            comp, survey.freqs
        )
    )
    # simulation.mapping = Maps.ExpMap(simulation.mesh)
    # simulation.sigmaPrimary = np.log(sigBG)
    x0 = np.log(sigBG)
    # cond = sig[0]
    # x0 = np.log(np.ones(simulation.mesh.nC)*cond)
    # simulation.sigmaPrimary = x0
    # if True:
    #     x0  = x0 + np.random.randn(simulation.mesh.nC)*cond*1e-1
    survey = simulation.survey

    def fun(x):
        return simulation.dpred(x), lambda x: simulation.Jvec(x0, x)

    return tests.checkDerivative(fun, x0, num=3, plotIt=False, eps=FLR)


def DerivProjfieldsTest(inputSetup, comp="All", freq=False):

    survey, simulation = nsem.utils.test_utils.setupSimpegNSEM_ePrimSec(
        inputSetup, comp, freq
    )
    print("Derivative test of data projection for eFormulation primary/secondary\n")
    # simulation.mapping = Maps.ExpMap(simulation.mesh)
    # Initate things for the derivs Test
    src = survey.source_list[0]
    np.random.seed(1983)
    u0x = np.random.randn(survey.mesh.nE) + np.random.randn(survey.mesh.nE) * 1j
    u0y = np.random.randn(survey.mesh.nE) + np.random.randn(survey.mesh.nE) * 1j
    u0 = np.vstack((mkvc(u0x, 2), mkvc(u0y, 2)))
    f0 = simulation.fieldsPair(survey.mesh, survey)
    # u0 = np.hstack((mkvc(u0_px,2),mkvc(u0_py,2)))
    f0[src, "e_pxSolution"] = u0[: len(u0) / 2]  # u0x
    f0[src, "e_pySolution"] = u0[len(u0) / 2 : :]  # u0y

    def fun(u):
        f = simulation.fieldsPair(survey.mesh, survey)
        f[src, "e_pxSolution"] = u[: len(u) / 2]
        f[src, "e_pySolution"] = u[len(u) / 2 : :]
        return (
            rx.eval(src, survey.mesh, f),
            lambda t: rx.evalDeriv(src, survey.mesh, f0, mkvc(t, 2)),
        )

    return tests.checkDerivative(fun, u0, num=3, plotIt=False, eps=FLR)


class NSEM_DerivTests(unittest.TestCase):
    def setUp(self):
        pass

    # Do a derivative test of Jvec
    def test_derivJvec_impedanceAll(self):
        self.assertTrue(
            DerivJvecTest(nsem.utils.test_utils.halfSpace(1e-2), "Imp", 0.1)
        )

    def test_derivJvec_zxxr(self):
        self.assertTrue(DerivJvecTest(nsem.utils.test_utils.halfSpace(1e-2), "xx", 0.1))

    def test_derivJvec_zxyi(self):
        self.assertTrue(DerivJvecTest(nsem.utils.test_utils.halfSpace(1e-2), "xy", 0.1))

    def test_derivJvec_zyxr(self):
        self.assertTrue(DerivJvecTest(nsem.utils.test_utils.halfSpace(1e-2), "yx", 0.1))

    def test_derivJvec_zyyi(self):
        self.assertTrue(DerivJvecTest(nsem.utils.test_utils.halfSpace(1e-2), "yy", 0.1))

    # Tipper
    def test_derivJvec_tipperAll(self):
        self.assertTrue(
            DerivJvecTest(nsem.utils.test_utils.halfSpace(1e-2), "Tip", 0.1)
        )

    def test_derivJvec_tzxr(self):
        self.assertTrue(DerivJvecTest(nsem.utils.test_utils.halfSpace(1e-2), "zx", 0.1))

    def test_derivJvec_tzyi(self):
        self.assertTrue(DerivJvecTest(nsem.utils.test_utils.halfSpace(1e-2), "zy", 0.1))


if __name__ == "__main__":
    unittest.main()
