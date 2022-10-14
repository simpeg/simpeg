from __future__ import print_function
import unittest
import numpy as np
from scipy.constants import mu_0
from SimPEG.electromagnetics.utils.testing_utils import getFDEMProblem

testE = True
testB = True

verbose = False

TOL = 1e-5
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = 1e-1
addrandoms = True

SrcList = ["RawVec", "MagDipole"]  # or 'MAgDipole_Bfield', 'CircularLoop', 'RawVec'


def adjointTest(fdemType, comp):
    prb = getFDEMProblem(fdemType, comp, SrcList, freq)
    # prb.solverOpts = dict(check_accuracy=True)
    print("Adjoint {0!s} formulation - {1!s}".format(fdemType, comp))

    m = np.log(np.ones(prb.sigmaMap.nP) * CONDUCTIVITY)
    mu = np.ones(prb.mesh.nC) * MU

    if addrandoms is True:
        m = m + np.random.randn(prb.sigmaMap.nP) * np.log(CONDUCTIVITY) * 1e-1
        mu = mu + np.random.randn(prb.mesh.nC) * MU * 1e-1

    survey = prb.survey
    # prb.PropMap.PropModel.mu = mu
    # prb.PropMap.PropModel.mui = 1./mu
    u = prb.fields(m)

    v = np.random.rand(survey.nD)
    w = np.random.rand(prb.mesh.nC)

    vJw = v.dot(prb.Jvec(m, w, u))
    wJtv = w.dot(prb.Jtvec(m, v, u))
    tol = np.max([TOL * (10 ** int(np.log10(np.abs(vJw)))), FLR])
    print(vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol)
    return np.abs(vJw - wJtv) < tol


class FDEM_AdjointTests(unittest.TestCase):
    if testE:

        def test_Jtvec_adjointTest_exr_Eform(self):
            self.assertTrue(adjointTest("e", ["ElectricField", "x", "r"]))

        def test_Jtvec_adjointTest_eyr_Eform(self):
            self.assertTrue(adjointTest("e", ["ElectricField", "y", "r"]))

        def test_Jtvec_adjointTest_ezr_Eform(self):
            self.assertTrue(adjointTest("e", ["ElectricField", "z", "r"]))

        def test_Jtvec_adjointTest_exi_Eform(self):
            self.assertTrue(adjointTest("e", ["ElectricField", "x", "i"]))

        def test_Jtvec_adjointTest_eyi_Eform(self):
            self.assertTrue(adjointTest("e", ["ElectricField", "y", "i"]))

        def test_Jtvec_adjointTest_ezi_Eform(self):
            self.assertTrue(adjointTest("e", ["ElectricField", "z", "i"]))

        def test_Jtvec_adjointTest_bxr_Eform(self):
            self.assertTrue(adjointTest("e", ["MagneticFluxDensity", "x", "r"]))

        def test_Jtvec_adjointTest_byr_Eform(self):
            self.assertTrue(adjointTest("e", ["MagneticFluxDensity", "y", "r"]))

        def test_Jtvec_adjointTest_bzr_Eform(self):
            self.assertTrue(adjointTest("e", ["MagneticFluxDensity", "z", "r"]))

        def test_Jtvec_adjointTest_bxi_Eform(self):
            self.assertTrue(adjointTest("e", ["MagneticFluxDensity", "x", "i"]))

        def test_Jtvec_adjointTest_byi_Eform(self):
            self.assertTrue(adjointTest("e", ["MagneticFluxDensity", "y", "i"]))

        def test_Jtvec_adjointTest_bzi_Eform(self):
            self.assertTrue(adjointTest("e", ["MagneticFluxDensity", "z", "i"]))

        def test_Jtvec_adjointTest_jxr_Eform(self):
            self.assertTrue(adjointTest("e", ["CurrentDensity", "x", "r"]))

        def test_Jtvec_adjointTest_jyr_Eform(self):
            self.assertTrue(adjointTest("e", ["CurrentDensity", "y", "r"]))

        def test_Jtvec_adjointTest_jzr_Eform(self):
            self.assertTrue(adjointTest("e", ["CurrentDensity", "z", "r"]))

        def test_Jtvec_adjointTest_jxi_Eform(self):
            self.assertTrue(adjointTest("e", ["CurrentDensity", "x", "i"]))

        def test_Jtvec_adjointTest_jyi_Eform(self):
            self.assertTrue(adjointTest("e", ["CurrentDensity", "y", "i"]))

        def test_Jtvec_adjointTest_jzi_Eform(self):
            self.assertTrue(adjointTest("e", ["CurrentDensity", "z", "i"]))

        def test_Jtvec_adjointTest_hxr_Eform(self):
            self.assertTrue(adjointTest("e", ["MagneticField", "x", "r"]))

        def test_Jtvec_adjointTest_hyr_Eform(self):
            self.assertTrue(adjointTest("e", ["MagneticField", "y", "r"]))

        def test_Jtvec_adjointTest_hzr_Eform(self):
            self.assertTrue(adjointTest("e", ["MagneticField", "z", "r"]))

        def test_Jtvec_adjointTest_hxi_Eform(self):
            self.assertTrue(adjointTest("e", ["MagneticField", "x", "i"]))

        def test_Jtvec_adjointTest_hyi_Eform(self):
            self.assertTrue(adjointTest("e", ["MagneticField", "y", "i"]))

        def test_Jtvec_adjointTest_hzi_Eform(self):
            self.assertTrue(adjointTest("e", ["MagneticField", "z", "i"]))

    if testB:

        def test_Jtvec_adjointTest_exr_Bform(self):
            self.assertTrue(adjointTest("b", ["ElectricField", "x", "r"]))

        def test_Jtvec_adjointTest_eyr_Bform(self):
            self.assertTrue(adjointTest("b", ["ElectricField", "y", "r"]))

        def test_Jtvec_adjointTest_ezr_Bform(self):
            self.assertTrue(adjointTest("b", ["ElectricField", "z", "r"]))

        def test_Jtvec_adjointTest_exi_Bform(self):
            self.assertTrue(adjointTest("b", ["ElectricField", "x", "i"]))

        def test_Jtvec_adjointTest_eyi_Bform(self):
            self.assertTrue(adjointTest("b", ["ElectricField", "y", "i"]))

        def test_Jtvec_adjointTest_ezi_Bform(self):
            self.assertTrue(adjointTest("b", ["ElectricField", "z", "i"]))

        def test_Jtvec_adjointTest_bxr_Bform(self):
            self.assertTrue(adjointTest("b", ["MagneticFluxDensity", "x", "r"]))

        def test_Jtvec_adjointTest_byr_Bform(self):
            self.assertTrue(adjointTest("b", ["MagneticFluxDensity", "y", "r"]))

        def test_Jtvec_adjointTest_bzr_Bform(self):
            self.assertTrue(adjointTest("b", ["MagneticFluxDensity", "z", "r"]))

        def test_Jtvec_adjointTest_bxi_Bform(self):
            self.assertTrue(adjointTest("b", ["MagneticFluxDensity", "x", "i"]))

        def test_Jtvec_adjointTest_byi_Bform(self):
            self.assertTrue(adjointTest("b", ["MagneticFluxDensity", "y", "i"]))

        def test_Jtvec_adjointTest_bzi_Bform(self):
            self.assertTrue(adjointTest("b", ["MagneticFluxDensity", "z", "i"]))

        def test_Jtvec_adjointTest_jxr_Bform(self):
            self.assertTrue(adjointTest("b", ["CurrentDensity", "x", "r"]))

        def test_Jtvec_adjointTest_jyr_Bform(self):
            self.assertTrue(adjointTest("b", ["CurrentDensity", "y", "r"]))

        def test_Jtvec_adjointTest_jzr_Bform(self):
            self.assertTrue(adjointTest("b", ["CurrentDensity", "z", "r"]))

        def test_Jtvec_adjointTest_jxi_Bform(self):
            self.assertTrue(adjointTest("b", ["CurrentDensity", "x", "i"]))

        def test_Jtvec_adjointTest_jyi_Bform(self):
            self.assertTrue(adjointTest("b", ["CurrentDensity", "y", "i"]))

        def test_Jtvec_adjointTest_jzi_Bform(self):
            self.assertTrue(adjointTest("b", ["CurrentDensity", "z", "i"]))

        def test_Jtvec_adjointTest_hxr_Bform(self):
            self.assertTrue(adjointTest("b", ["MagneticField", "x", "r"]))

        def test_Jtvec_adjointTest_hyr_Bform(self):
            self.assertTrue(adjointTest("b", ["MagneticField", "y", "r"]))

        def test_Jtvec_adjointTest_hzr_Bform(self):
            self.assertTrue(adjointTest("b", ["MagneticField", "z", "r"]))

        def test_Jtvec_adjointTest_hxi_Bform(self):
            self.assertTrue(adjointTest("b", ["MagneticField", "x", "i"]))

        def test_Jtvec_adjointTest_hyi_Bform(self):
            self.assertTrue(adjointTest("b", ["MagneticField", "y", "i"]))

        def test_Jtvec_adjointTest_hzi_Bform(self):
            self.assertTrue(adjointTest("b", ["MagneticField", "z", "i"]))
