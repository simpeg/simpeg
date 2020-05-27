from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import unittest
import numpy as np
from scipy.constants import mu_0
from SimPEG import maps, mkvc, tests
from SimPEG.electromagnetics import natural_source as nsem

TOL = 1e-4
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0


def DerivJvecTest(halfspace_value, freq=False, expMap=True):

    survey, sig, sigBG, mesh = nsem.utils.test_utils.setup1DSurvey(
        halfspace_value, False, structure=True
    )
    simulation = nsem.Simulation1DPrimarySecondary(
        mesh, sigmaPrimary=sigBG, sigmaMap=maps.IdentityMap(mesh), survey=survey
    )
    print("Using {0} solver for the simulation".format(simulation.Solver))
    print(
        "Derivative test of Jvec for eForm primary/secondary for 1d comp from {0} to {1} Hz\n".format(
            survey.freqs[0], survey.freqs[-1]
        )
    )
    # simulation.mapping = maps.ExpMap(simulation.mesh)
    # simulation.sigmaPrimary = np.log(sigBG)

    x0 = sigBG
    # cond = sig[0]
    # x0 = np.log(np.ones(simulation.mesh.nC)*halfspace_value)
    # simulation.sigmaPrimary = x0
    np.random.seed(1983)
    # if True:
    #     x0  = x0 + np.random.randn(simulation.mesh.nC)*halfspace_value*1e-1
    survey = simulation.survey

    def fun(x):
        return simulation.dpred(x), lambda x: simulation.Jvec(x0, x)

    return tests.checkDerivative(fun, x0, num=4, plotIt=False, eps=FLR)


def DerivProjfieldsTest(inputSetup, comp="All", freq=False):

    survey, simulation = nsem.utils.test_utils.setupSimpegNSEM_ePrimSec(
        inputSetup, comp, freq
    )
    print("Derivative test of data projection for eFormulation primary/secondary\n")
    # simulation.mapping = maps.ExpMap(simulation.mesh)
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

    return tests.checkDerivative(fun, u0, num=4, plotIt=False, eps=FLR)


class NSEM_DerivTests(unittest.TestCase):
    def test_derivJvec_Z1dr(self):
        self.assertTrue(DerivJvecTest(1e-2))

    def test_derivJvec_Z1di(self):
        self.assertTrue(DerivJvecTest(1e-2))


if __name__ == "__main__":
    unittest.main()
