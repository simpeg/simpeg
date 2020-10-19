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

SrcType = ["MagDipole", "RawVec"]  # or 'MAgDipole_Bfield', 'CircularLoop', 'RawVec'


def derivTest(fdemType, comp):

    prb = getFDEMProblem(fdemType, comp, SrcType, freq)
    # prb.solverOpts = dict(check_accuracy=True)

    print("{0!s} formulation - {1!s}".format(fdemType, comp))
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

        def test_Jvec_exr_Eform(self):
            self.assertTrue(derivTest("e", ["ElectricField", "x", "r"]))

        def test_Jvec_eyr_Eform(self):
            self.assertTrue(derivTest("e", ["ElectricField", "y", "r"]))

        def test_Jvec_ezr_Eform(self):
            self.assertTrue(derivTest("e", ["ElectricField", "z", "r"]))

        def test_Jvec_exi_Eform(self):
            self.assertTrue(derivTest("e", ["ElectricField", "x", "i"]))

        def test_Jvec_eyi_Eform(self):
            self.assertTrue(derivTest("e", ["ElectricField", "y", "i"]))

        def test_Jvec_ezi_Eform(self):
            self.assertTrue(derivTest("e", ["ElectricField", "z", "i"]))

        def test_Jvec_bxr_Eform(self):
            self.assertTrue(derivTest("e", ["MagneticFluxDensity", "x", "r"]))

        def test_Jvec_byr_Eform(self):
            self.assertTrue(derivTest("e", ["MagneticFluxDensity", "y", "r"]))

        def test_Jvec_bzr_Eform(self):
            self.assertTrue(derivTest("e", ["MagneticFluxDensity", "z", "r"]))

        def test_Jvec_bxi_Eform(self):
            self.assertTrue(derivTest("e", ["MagneticFluxDensity", "x", "i"]))

        def test_Jvec_byi_Eform(self):
            self.assertTrue(derivTest("e", ["MagneticFluxDensity", "y", "i"]))

        def test_Jvec_bzi_Eform(self):
            self.assertTrue(derivTest("e", ["MagneticFluxDensity", "z", "i"]))

        def test_Jvec_exr_Eform(self):
            self.assertTrue(derivTest("e", ["CurrentDensity", "x", "r"]))

        def test_Jvec_eyr_Eform(self):
            self.assertTrue(derivTest("e", ["CurrentDensity", "y", "r"]))

        def test_Jvec_ezr_Eform(self):
            self.assertTrue(derivTest("e", ["CurrentDensity", "z", "r"]))

        def test_Jvec_exi_Eform(self):
            self.assertTrue(derivTest("e", ["CurrentDensity", "x", "i"]))

        def test_Jvec_eyi_Eform(self):
            self.assertTrue(derivTest("e", ["CurrentDensity", "y", "i"]))

        def test_Jvec_ezi_Eform(self):
            self.assertTrue(derivTest("e", ["CurrentDensity", "z", "i"]))

        def test_Jvec_bxr_Eform(self):
            self.assertTrue(derivTest("e", ["MagneticField", "x", "r"]))

        def test_Jvec_byr_Eform(self):
            self.assertTrue(derivTest("e", ["MagneticField", "y", "r"]))

        def test_Jvec_bzr_Eform(self):
            self.assertTrue(derivTest("e", ["MagneticField", "z", "r"]))

        def test_Jvec_bxi_Eform(self):
            self.assertTrue(derivTest("e", ["MagneticField", "x", "i"]))

        def test_Jvec_byi_Eform(self):
            self.assertTrue(derivTest("e", ["MagneticField", "y", "i"]))

        def test_Jvec_bzi_Eform(self):
            self.assertTrue(derivTest("e", ["MagneticField", "z", "i"]))

    if testB:

        def test_Jvec_exr_Bform(self):
            self.assertTrue(derivTest("b", ["ElectricField", "x", "r"]))

        def test_Jvec_eyr_Bform(self):
            self.assertTrue(derivTest("b", ["ElectricField", "y", "r"]))

        def test_Jvec_ezr_Bform(self):
            self.assertTrue(derivTest("b", ["ElectricField", "z", "r"]))

        def test_Jvec_exi_Bform(self):
            self.assertTrue(derivTest("b", ["ElectricField", "x", "i"]))

        def test_Jvec_eyi_Bform(self):
            self.assertTrue(derivTest("b", ["ElectricField", "y", "i"]))

        def test_Jvec_ezi_Bform(self):
            self.assertTrue(derivTest("b", ["ElectricField", "z", "i"]))

        def test_Jvec_bxr_Bform(self):
            self.assertTrue(derivTest("b", ["MagneticFluxDensity", "x", "r"]))

        def test_Jvec_byr_Bform(self):
            self.assertTrue(derivTest("b", ["MagneticFluxDensity", "y", "r"]))

        def test_Jvec_bzr_Bform(self):
            self.assertTrue(derivTest("b", ["MagneticFluxDensity", "z", "r"]))

        def test_Jvec_bxi_Bform(self):
            self.assertTrue(derivTest("b", ["MagneticFluxDensity", "x", "i"]))

        def test_Jvec_byi_Bform(self):
            self.assertTrue(derivTest("b", ["MagneticFluxDensity", "y", "i"]))

        def test_Jvec_bzi_Bform(self):
            self.assertTrue(derivTest("b", ["MagneticFluxDensity", "z", "i"]))

        def test_Jvec_jxr_Bform(self):
            self.assertTrue(derivTest("b", ["CurrentDensity", "x", "r"]))

        def test_Jvec_jyr_Bform(self):
            self.assertTrue(derivTest("b", ["CurrentDensity", "y", "r"]))

        def test_Jvec_jzr_Bform(self):
            self.assertTrue(derivTest("b", ["CurrentDensity", "z", "r"]))

        def test_Jvec_jxi_Bform(self):
            self.assertTrue(derivTest("b", ["CurrentDensity", "x", "i"]))

        def test_Jvec_jyi_Bform(self):
            self.assertTrue(derivTest("b", ["CurrentDensity", "y", "i"]))

        def test_Jvec_jzi_Bform(self):
            self.assertTrue(derivTest("b", ["CurrentDensity", "z", "i"]))

        def test_Jvec_hxr_Bform(self):
            self.assertTrue(derivTest("b", ["MagneticField", "x", "r"]))

        def test_Jvec_hyr_Bform(self):
            self.assertTrue(derivTest("b", ["MagneticField", "y", "r"]))

        def test_Jvec_hzr_Bform(self):
            self.assertTrue(derivTest("b", ["MagneticField", "z", "r"]))

        def test_Jvec_hxi_Bform(self):
            self.assertTrue(derivTest("b", ["MagneticField", "x", "i"]))

        def test_Jvec_hyi_Bform(self):
            self.assertTrue(derivTest("b", ["MagneticField", "y", "i"]))

        def test_Jvec_hzi_Bform(self):
            self.assertTrue(derivTest("b", ["MagneticField", "z", "i"]))

    if testJ:

        def test_Jvec_jxr_Jform(self):
            self.assertTrue(derivTest("j", ["CurrentDensity", "x", "r"]))

        def test_Jvec_jyr_Jform(self):
            self.assertTrue(derivTest("j", ["CurrentDensity", "y", "r"]))

        def test_Jvec_jzr_Jform(self):
            self.assertTrue(derivTest("j", ["CurrentDensity", "z", "r"]))

        def test_Jvec_jxi_Jform(self):
            self.assertTrue(derivTest("j", ["CurrentDensity", "x", "i"]))

        def test_Jvec_jyi_Jform(self):
            self.assertTrue(derivTest("j", ["CurrentDensity", "y", "i"]))

        def test_Jvec_jzi_Jform(self):
            self.assertTrue(derivTest("j", ["CurrentDensity", "z", "i"]))

        def test_Jvec_hxr_Jform(self):
            self.assertTrue(derivTest("j", ["MagneticField", "x", "r"]))

        def test_Jvec_hyr_Jform(self):
            self.assertTrue(derivTest("j", ["MagneticField", "y", "r"]))

        def test_Jvec_hzr_Jform(self):
            self.assertTrue(derivTest("j", ["MagneticField", "z", "r"]))

        def test_Jvec_hxi_Jform(self):
            self.assertTrue(derivTest("j", ["MagneticField", "x", "i"]))

        def test_Jvec_hyi_Jform(self):
            self.assertTrue(derivTest("j", ["MagneticField", "y", "i"]))

        def test_Jvec_hzi_Jform(self):
            self.assertTrue(derivTest("j", ["MagneticField", "z", "i"]))

        def test_Jvec_exr_Jform(self):
            self.assertTrue(derivTest("j", ["ElectricField", "x", "r"]))

        def test_Jvec_eyr_Jform(self):
            self.assertTrue(derivTest("j", ["ElectricField", "y", "r"]))

        def test_Jvec_ezr_Jform(self):
            self.assertTrue(derivTest("j", ["ElectricField", "z", "r"]))

        def test_Jvec_exi_Jform(self):
            self.assertTrue(derivTest("j", ["ElectricField", "x", "i"]))

        def test_Jvec_eyi_Jform(self):
            self.assertTrue(derivTest("j", ["ElectricField", "y", "i"]))

        def test_Jvec_ezi_Jform(self):
            self.assertTrue(derivTest("j", ["ElectricField", "z", "i"]))

        def test_Jvec_bxr_Jform(self):
            self.assertTrue(derivTest("j", ["MagneticFluxDensity", "x", "r"]))

        def test_Jvec_byr_Jform(self):
            self.assertTrue(derivTest("j", ["MagneticFluxDensity", "y", "r"]))

        def test_Jvec_bzr_Jform(self):
            self.assertTrue(derivTest("j", ["MagneticFluxDensity", "z", "r"]))

        def test_Jvec_bxi_Jform(self):
            self.assertTrue(derivTest("j", ["MagneticFluxDensity", "x", "i"]))

        def test_Jvec_byi_Jform(self):
            self.assertTrue(derivTest("j", ["MagneticFluxDensity", "y", "i"]))

        def test_Jvec_bzi_Jform(self):
            self.assertTrue(derivTest("j", ["MagneticFluxDensity", "z", "i"]))

    if testH:

        def test_Jvec_hxr_Hform(self):
            self.assertTrue(derivTest("h", ["MagneticField", "x", "r"]))

        def test_Jvec_hyr_Hform(self):
            self.assertTrue(derivTest("h", ["MagneticField", "y", "r"]))

        def test_Jvec_hzr_Hform(self):
            self.assertTrue(derivTest("h", ["MagneticField", "z", "r"]))

        def test_Jvec_hxi_Hform(self):
            self.assertTrue(derivTest("h", ["MagneticField", "x", "i"]))

        def test_Jvec_hyi_Hform(self):
            self.assertTrue(derivTest("h", ["MagneticField", "y", "i"]))

        def test_Jvec_hzi_Hform(self):
            self.assertTrue(derivTest("h", ["MagneticField", "z", "i"]))

        def test_Jvec_hxr_Hform(self):
            self.assertTrue(derivTest("h", ["CurrentDensity", "x", "r"]))

        def test_Jvec_hyr_Hform(self):
            self.assertTrue(derivTest("h", ["CurrentDensity", "y", "r"]))

        def test_Jvec_hzr_Hform(self):
            self.assertTrue(derivTest("h", ["CurrentDensity", "z", "r"]))

        def test_Jvec_hxi_Hform(self):
            self.assertTrue(derivTest("h", ["CurrentDensity", "x", "i"]))

        def test_Jvec_hyi_Hform(self):
            self.assertTrue(derivTest("h", ["CurrentDensity", "y", "i"]))

        def test_Jvec_hzi_Hform(self):
            self.assertTrue(derivTest("h", ["CurrentDensity", "z", "i"]))

        def test_Jvec_exr_Hform(self):
            self.assertTrue(derivTest("h", ["ElectricField", "x", "r"]))

        def test_Jvec_eyr_Hform(self):
            self.assertTrue(derivTest("h", ["ElectricField", "y", "r"]))

        def test_Jvec_ezr_Hform(self):
            self.assertTrue(derivTest("h", ["ElectricField", "z", "r"]))

        def test_Jvec_exi_Hform(self):
            self.assertTrue(derivTest("h", ["ElectricField", "x", "i"]))

        def test_Jvec_eyi_Hform(self):
            self.assertTrue(derivTest("h", ["ElectricField", "y", "i"]))

        def test_Jvec_ezi_Hform(self):
            self.assertTrue(derivTest("h", ["ElectricField", "z", "i"]))

        def test_Jvec_bxr_Hform(self):
            self.assertTrue(derivTest("h", ["MagneticFluxDensity", "x", "r"]))

        def test_Jvec_byr_Hform(self):
            self.assertTrue(derivTest("h", ["MagneticFluxDensity", "y", "r"]))

        def test_Jvec_bzr_Hform(self):
            self.assertTrue(derivTest("h", ["MagneticFluxDensity", "z", "r"]))

        def test_Jvec_bxi_Hform(self):
            self.assertTrue(derivTest("h", ["MagneticFluxDensity", "x", "i"]))

        def test_Jvec_byi_Hform(self):
            self.assertTrue(derivTest("h", ["MagneticFluxDensity", "y", "i"]))

        def test_Jvec_bzi_Hform(self):
            self.assertTrue(derivTest("h", ["MagneticFluxDensity", "z", "i"]))


if __name__ == "__main__":
    unittest.main()
