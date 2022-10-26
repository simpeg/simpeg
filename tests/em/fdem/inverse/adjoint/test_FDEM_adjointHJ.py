from __future__ import print_function
import unittest
import numpy as np
from scipy.constants import mu_0
from SimPEG.electromagnetics.utils.testing_utils import getFDEMProblem

testJ = True
testH = True

verbose = False

TOL = 1e-5
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = 1e-1
addrandoms = True

SrcList = ["RawVec", "MagDipole", "MagDipole_Bfield", "CircularLoop", "LineCurrent"]


def adjointTest(fdemType, comp, src):
    prb = getFDEMProblem(fdemType, comp, [src], freq)
    # prb.solverOpts = dict(check_accuracy=True)
    print(f"Adjoint {fdemType} formulation - {src} - {comp}")

    m = np.log(np.ones(prb.sigmaMap.nP) * CONDUCTIVITY)
    mu = np.ones(prb.mesh.nC) * MU

    if addrandoms is True:
        m = m + np.random.randn(prb.sigmaMap.nP) * np.log(CONDUCTIVITY) * 1e-1
        mu = mu + np.random.randn(prb.mesh.nC) * MU * 1e-1

    survey = prb.survey
    u = prb.fields(m)

    v = np.random.rand(survey.nD)
    w = np.random.rand(prb.mesh.nC)

    vJw = v.dot(prb.Jvec(m, w, u))
    wJtv = w.dot(prb.Jtvec(m, v, u))
    tol = np.max([TOL * (10 ** int(np.log10(np.abs(vJw)))), FLR])
    print(vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol)
    return np.abs(vJw - wJtv) < tol


class FDEM_AdjointTests(unittest.TestCase):

    if testJ:

        def test_Jtvec_adjointTest_j_Jform(self):
            for src in SrcList:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            adjointTest("j", ["CurrentDensity", orientation, comp], src)
                        )

        def test_Jtvec_adjointTest_h_Jform(self):
            for src in SrcList:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            adjointTest("j", ["MagneticField", orientation, comp], src)
                        )

        def test_Jtvec_adjointTest_exr_Jform(self):
            for src in SrcList:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            adjointTest("j", ["ElectricField", orientation, comp], src)
                        )

        def test_Jtvec_adjointTest_bxr_Jform(self):
            for src in SrcList:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            adjointTest(
                                "j", ["MagneticFluxDensity", orientation, comp], src
                            )
                        )

    if testH:

        def test_Jtvec_adjointTest_j_Hform(self):
            for src in SrcList:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            adjointTest("h", ["CurrentDensity", orientation, comp], src)
                        )

        def test_Jtvec_adjointTest_h_Hform(self):
            for src in SrcList:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            adjointTest("h", ["MagneticField", orientation, comp], src)
                        )

        def test_Jtvec_adjointTest_exr_Hform(self):
            for src in SrcList:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            adjointTest("h", ["ElectricField", orientation, comp], src)
                        )

        def test_Jtvec_adjointTest_bxr_Hform(self):
            for src in SrcList:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            adjointTest(
                                "h", ["MagneticFluxDensity", orientation, comp], src
                            )
                        )
