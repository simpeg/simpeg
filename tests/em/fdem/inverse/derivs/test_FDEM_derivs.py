from __future__ import print_function
import unittest
import numpy as np
from SimPEG import tests
from scipy.constants import mu_0
from SimPEG.electromagnetics.utils.testing_utils import getFDEMProblem

testE = False
testB = True
testH = False
testJ = False

verbose = False

TOL = 1e-5
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = 3.16
addrandoms = True

SrcType = [
    "MagDipole",
    "CircularLoop",
    "MagDipole_Bfield",
    "RawVec",
    "LineCurrent",
]  # or 'MAgDipole_Bfield', 'CircularLoop', 'RawVec'


def derivTest(fdemType, comp, src):

    prb = getFDEMProblem(fdemType, comp, SrcType, freq)
    # prb.solverOpts = dict(check_accuracy=True)

    print(f"{fdemType} formulation {src} - {comp}")
    x0 = np.log(np.ones(prb.sigmaMap.nP) * CONDUCTIVITY)
    # mu = np.log(np.ones(prb.mesh.nC)*MU)

    if addrandoms is True:
        x0 = x0 + np.random.randn(prb.sigmaMap.nP) * np.log(CONDUCTIVITY) * 1e-1
        # mu = mu + np.random.randn(prb.sigmaMap.nP)*MU*1e-1

    survey = prb.survey

    def fun(x):
        return prb.dpred(x), lambda x: prb.Jvec(x0, x)

    return tests.checkDerivative(fun, x0, num=2, plotIt=False, eps=FLR)


class FDEM_DerivTests(unittest.TestCase):
    if testE:

        def test_Jvec_e_Eform(self):
            for src in SrcType:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            derivTest("e", ["ElectricField", orientation, comp], src)
                        )

        def test_Jvec_b_Eform(self):
            for src in SrcType:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            derivTest(
                                "e", ["MagneticFluxDensity", orientation, comp], src
                            )
                        )

        def test_Jvec_j_Eform(self):
            for src in SrcType:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            derivTest("e", ["CurrentDensity", orientation, comp], src)
                        )

        def test_Jvec_h_Eform(self):
            for src in SrcType:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            derivTest("e", ["MagneticField", orientation, comp], src)
                        )

    if testB:

        def test_Jvec_e_Bform(self):
            for src in SrcType:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            derivTest("b", ["ElectricField", orientation, comp], src)
                        )

        def test_Jvec_b_Bform(self):
            for src in SrcType:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            derivTest(
                                "b", ["MagneticFluxDensity", orientation, comp], src
                            )
                        )

        def test_Jvec_j_Bform(self):
            for src in SrcType:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            derivTest("b", ["CurrentDensity", orientation, comp], src)
                        )

        def test_Jvec_h_Bform(self):
            for src in SrcType:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            derivTest("b", ["MagneticField", orientation, comp], src)
                        )

    if testJ:

        def test_Jvec_e_Jform(self):
            for src in SrcType:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            derivTest("j", ["ElectricField", orientation, comp], src)
                        )

        def test_Jvec_b_Jform(self):
            for src in SrcType:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            derivTest(
                                "j", ["MagneticFluxDensity", orientation, comp], src
                            )
                        )

        def test_Jvec_j_Jform(self):
            for src in SrcType:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            derivTest("j", ["CurrentDensity", orientation, comp], src)
                        )

        def test_Jvec_h_Jform(self):
            for src in SrcType:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            derivTest("j", ["MagneticField", orientation, comp], src)
                        )

    if testH:

        def test_Jvec_e_Hform(self):
            for src in SrcType:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            derivTest("h", ["ElectricField", orientation, comp], src)
                        )

        def test_Jvec_b_Hform(self):
            for src in SrcType:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            derivTest(
                                "h", ["MagneticFluxDensity", orientation, comp], src
                            )
                        )

        def test_Jvec_j_Hform(self):
            for src in SrcType:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            derivTest("h", ["CurrentDensity", orientation, comp], src)
                        )

        def test_Jvec_h_Hform(self):
            for src in SrcType:
                for orientation in ["x", "y", "z"]:
                    for comp in ["r", "i"]:
                        self.assertTrue(
                            derivTest("h", ["MagneticField", orientation, comp], src)
                        )
