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

        def test_Jtvec_adjointTest_jxr_Jform(self):
            self.assertTrue(adjointTest("j", ["CurrentDensity", "x", "r"]))

        def test_Jtvec_adjointTest_jyr_Jform(self):
            self.assertTrue(adjointTest("j", ["CurrentDensity", "y", "r"]))

        def test_Jtvec_adjointTest_jzr_Jform(self):
            self.assertTrue(adjointTest("j", ["CurrentDensity", "z", "r"]))

        def test_Jtvec_adjointTest_jxi_Jform(self):
            self.assertTrue(adjointTest("j", ["CurrentDensity", "x", "i"]))

        def test_Jtvec_adjointTest_jyi_Jform(self):
            self.assertTrue(adjointTest("j", ["CurrentDensity", "y", "i"]))

        def test_Jtvec_adjointTest_jzi_Jform(self):
            self.assertTrue(adjointTest("j", ["CurrentDensity", "z", "i"]))

        def test_Jtvec_adjointTest_hxr_Jform(self):
            self.assertTrue(adjointTest("j", ["MagneticField", "x", "r"]))

        def test_Jtvec_adjointTest_hyr_Jform(self):
            self.assertTrue(adjointTest("j", ["MagneticField", "y", "r"]))

        def test_Jtvec_adjointTest_hzr_Jform(self):
            self.assertTrue(adjointTest("j", ["MagneticField", "z", "r"]))

        def test_Jtvec_adjointTest_hxi_Jform(self):
            self.assertTrue(adjointTest("j", ["MagneticField", "x", "i"]))

        def test_Jtvec_adjointTest_hyi_Jform(self):
            self.assertTrue(adjointTest("j", ["MagneticField", "y", "i"]))

        def test_Jtvec_adjointTest_hzi_Jform(self):
            self.assertTrue(adjointTest("j", ["MagneticField", "z", "i"]))

        def test_Jtvec_adjointTest_exr_Jform(self):
            self.assertTrue(adjointTest("j", ["ElectricField", "x", "r"]))

        def test_Jtvec_adjointTest_eyr_Jform(self):
            self.assertTrue(adjointTest("j", ["ElectricField", "y", "r"]))

        def test_Jtvec_adjointTest_ezr_Jform(self):
            self.assertTrue(adjointTest("j", ["ElectricField", "z", "r"]))

        def test_Jtvec_adjointTest_exi_Jform(self):
            self.assertTrue(adjointTest("j", ["ElectricField", "x", "i"]))

        def test_Jtvec_adjointTest_eyi_Jform(self):
            self.assertTrue(adjointTest("j", ["ElectricField", "y", "i"]))

        def test_Jtvec_adjointTest_ezi_Jform(self):
            self.assertTrue(adjointTest("j", ["ElectricField", "z", "i"]))

        def test_Jtvec_adjointTest_bxr_Jform(self):
            self.assertTrue(adjointTest("j", ["MagneticFluxDensity", "x", "r"]))

        def test_Jtvec_adjointTest_byr_Jform(self):
            self.assertTrue(adjointTest("j", ["MagneticFluxDensity", "y", "r"]))

        def test_Jtvec_adjointTest_bzr_Jform(self):
            self.assertTrue(adjointTest("j", ["MagneticFluxDensity", "z", "r"]))

        def test_Jtvec_adjointTest_bxi_Jform(self):
            self.assertTrue(adjointTest("j", ["MagneticFluxDensity", "x", "i"]))

        def test_Jtvec_adjointTest_byi_Jform(self):
            self.assertTrue(adjointTest("j", ["MagneticFluxDensity", "y", "i"]))

        def test_Jtvec_adjointTest_bzi_Jform(self):
            self.assertTrue(adjointTest("j", ["MagneticFluxDensity", "z", "i"]))

    if testH:

        def test_Jtvec_adjointTest_hxr_Hform(self):
            self.assertTrue(adjointTest("h", ["MagneticField", "x", "r"]))

        def test_Jtvec_adjointTest_hyr_Hform(self):
            self.assertTrue(adjointTest("h", ["MagneticField", "y", "r"]))

        def test_Jtvec_adjointTest_hzr_Hform(self):
            self.assertTrue(adjointTest("h", ["MagneticField", "z", "r"]))

        def test_Jtvec_adjointTest_hxi_Hform(self):
            self.assertTrue(adjointTest("h", ["MagneticField", "x", "i"]))

        def test_Jtvec_adjointTest_hyi_Hform(self):
            self.assertTrue(adjointTest("h", ["MagneticField", "y", "i"]))

        def test_Jtvec_adjointTest_hzi_Hform(self):
            self.assertTrue(adjointTest("h", ["MagneticField", "z", "i"]))

        def test_Jtvec_adjointTest_jxr_Hform(self):
            self.assertTrue(adjointTest("h", ["CurrentDensity", "x", "r"]))

        def test_Jtvec_adjointTest_jyr_Hform(self):
            self.assertTrue(adjointTest("h", ["CurrentDensity", "y", "r"]))

        def test_Jtvec_adjointTest_jzr_Hform(self):
            self.assertTrue(adjointTest("h", ["CurrentDensity", "z", "r"]))

        def test_Jtvec_adjointTest_jxi_Hform(self):
            self.assertTrue(adjointTest("h", ["CurrentDensity", "x", "i"]))

        def test_Jtvec_adjointTest_jyi_Hform(self):
            self.assertTrue(adjointTest("h", ["CurrentDensity", "y", "i"]))

        def test_Jtvec_adjointTest_jzi_Hform(self):
            self.assertTrue(adjointTest("h", ["CurrentDensity", "z", "i"]))

        def test_Jtvec_adjointTest_exr_Hform(self):
            self.assertTrue(adjointTest("h", ["ElectricField", "x", "r"]))

        def test_Jtvec_adjointTest_eyr_Hform(self):
            self.assertTrue(adjointTest("h", ["ElectricField", "y", "r"]))

        def test_Jtvec_adjointTest_ezr_Hform(self):
            self.assertTrue(adjointTest("h", ["ElectricField", "z", "r"]))

        def test_Jtvec_adjointTest_exi_Hform(self):
            self.assertTrue(adjointTest("h", ["ElectricField", "x", "i"]))

        def test_Jtvec_adjointTest_eyi_Hform(self):
            self.assertTrue(adjointTest("h", ["ElectricField", "y", "i"]))

        def test_Jtvec_adjointTest_ezi_Hform(self):
            self.assertTrue(adjointTest("h", ["ElectricField", "z", "i"]))

        def test_Jtvec_adjointTest_bxr_Hform(self):
            self.assertTrue(adjointTest("h", ["MagneticFluxDensity", "x", "r"]))

        def test_Jtvec_adjointTest_byr_Hform(self):
            self.assertTrue(adjointTest("h", ["MagneticFluxDensity", "y", "r"]))

        def test_Jtvec_adjointTest_bzr_Hform(self):
            self.assertTrue(adjointTest("h", ["MagneticFluxDensity", "z", "r"]))

        def test_Jtvec_adjointTest_bxi_Hform(self):
            self.assertTrue(adjointTest("h", ["MagneticFluxDensity", "x", "i"]))

        def test_Jtvec_adjointTest_byi_Hform(self):
            self.assertTrue(adjointTest("h", ["MagneticFluxDensity", "y", "i"]))

        def test_Jtvec_adjointTest_bzi_Hform(self):
            self.assertTrue(adjointTest("h", ["MagneticFluxDensity", "z", "i"]))


if __name__ == "__main__":
    unittest.main()
