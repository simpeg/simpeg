import unittest
import numpy as np
from scipy.constants import mu_0
from SimPEG.electromagnetics.utils.testing_utils import (
    getFDEMProblem,
    getFDEMProblem_FaceEdgeConductivity,
)

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


def adjointTest(fdemType, comp, sigma_only=True):
    if sigma_only:
        prb = getFDEMProblem(fdemType, comp, SrcList, freq)
    else:
        prb = getFDEMProblem_FaceEdgeConductivity(fdemType, comp, SrcList, freq)
    # prb.solverOpts = dict(check_accuracy=True)
    print("Adjoint {0!s} formulation - {1!s}".format(fdemType, comp))

    m = np.log(
        np.ones(prb.sigmaMap.nP) * CONDUCTIVITY
    )  # works for sigma_only and sigma, tau, kappa
    mu = np.ones(prb.mesh.nC) * MU

    if addrandoms is True:
        m = m + np.random.randn(prb.sigmaMap.nP) * np.log(CONDUCTIVITY) * 1e-1
        mu = mu + np.random.randn(prb.mesh.nC) * MU * 1e-1

    survey = prb.survey
    # prb.PropMap.PropModel.mu = mu
    # prb.PropMap.PropModel.mui = 1./mu
    u = prb.fields(m)

    v = np.random.rand(survey.nD)
    w = np.random.rand(prb.sigmaMap.nP)  # works for sigma_only and sigma, tau, kappa

    vJw = v.dot(prb.Jvec(m, w, u))
    wJtv = w.dot(prb.Jtvec(m, v, u))
    tol = np.max([TOL * (10 ** int(np.log10(np.abs(vJw)))), FLR])
    print(vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol)
    return np.abs(vJw - wJtv) < tol


class FDEM_AdjointTests(unittest.TestCase):
    if testE:
        # SIGMA ONLY
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

        # FACE EDGE CONDUCTIVITY
        def test_Jtvec_adjointTest_exr_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["ElectricField", "x", "r"], False))

        def test_Jtvec_adjointTest_eyr_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["ElectricField", "y", "r"], False))

        def test_Jtvec_adjointTest_ezr_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["ElectricField", "z", "r"], False))

        def test_Jtvec_adjointTest_exi_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["ElectricField", "x", "i"], False))

        def test_Jtvec_adjointTest_eyi_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["ElectricField", "y", "i"], False))

        def test_Jtvec_adjointTest_ezi_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["ElectricField", "z", "i"], False))

        def test_Jtvec_adjointTest_bxr_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["MagneticFluxDensity", "x", "r"], False))

        def test_Jtvec_adjointTest_byr_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["MagneticFluxDensity", "y", "r"], False))

        def test_Jtvec_adjointTest_bzr_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["MagneticFluxDensity", "z", "r"], False))

        def test_Jtvec_adjointTest_bxi_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["MagneticFluxDensity", "x", "i"], False))

        def test_Jtvec_adjointTest_byi_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["MagneticFluxDensity", "y", "i"], False))

        def test_Jtvec_adjointTest_bzi_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["MagneticFluxDensity", "z", "i"], False))

        def test_Jtvec_adjointTest_jxr_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["CurrentDensity", "x", "r"], False))

        def test_Jtvec_adjointTest_jyr_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["CurrentDensity", "y", "r"], False))

        def test_Jtvec_adjointTest_jzr_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["CurrentDensity", "z", "r"], False))

        def test_Jtvec_adjointTest_jxi_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["CurrentDensity", "x", "i"], False))

        def test_Jtvec_adjointTest_jyi_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["CurrentDensity", "y", "i"], False))

        def test_Jtvec_adjointTest_jzi_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["CurrentDensity", "z", "i"], False))

        def test_Jtvec_adjointTest_hxr_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["MagneticField", "x", "r"], False))

        def test_Jtvec_adjointTest_hyr_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["MagneticField", "y", "r"], False))

        def test_Jtvec_adjointTest_hzr_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["MagneticField", "z", "r"], False))

        def test_Jtvec_adjointTest_hxi_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["MagneticField", "x", "i"], False))

        def test_Jtvec_adjointTest_hyi_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["MagneticField", "y", "i"], False))

        def test_Jtvec_adjointTest_hzi_Eform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("e", ["MagneticField", "z", "i"], False))

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

        # FACE EDGE CONDUCTIVITY
        def test_Jtvec_adjointTest_exr_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["ElectricField", "x", "r"], False))

        def test_Jtvec_adjointTest_eyr_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["ElectricField", "y", "r"], False))

        def test_Jtvec_adjointTest_ezr_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["ElectricField", "z", "r"], False))

        def test_Jtvec_adjointTest_exi_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["ElectricField", "x", "i"], False))

        def test_Jtvec_adjointTest_eyi_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["ElectricField", "y", "i"], False))

        def test_Jtvec_adjointTest_ezi_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["ElectricField", "z", "i"], False))

        def test_Jtvec_adjointTest_bxr_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["MagneticFluxDensity", "x", "r"], False))

        def test_Jtvec_adjointTest_byr_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["MagneticFluxDensity", "y", "r"], False))

        def test_Jtvec_adjointTest_bzr_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["MagneticFluxDensity", "z", "r"], False))

        def test_Jtvec_adjointTest_bxi_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["MagneticFluxDensity", "x", "i"], False))

        def test_Jtvec_adjointTest_byi_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["MagneticFluxDensity", "y", "i"], False))

        def test_Jtvec_adjointTest_bzi_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["MagneticFluxDensity", "z", "i"], False))

        def test_Jtvec_adjointTest_jxr_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["CurrentDensity", "x", "r"], False))

        def test_Jtvec_adjointTest_jyr_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["CurrentDensity", "y", "r"], False))

        def test_Jtvec_adjointTest_jzr_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["CurrentDensity", "z", "r"], False))

        def test_Jtvec_adjointTest_jxi_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["CurrentDensity", "x", "i"], False))

        def test_Jtvec_adjointTest_jyi_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["CurrentDensity", "y", "i"], False))

        def test_Jtvec_adjointTest_jzi_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["CurrentDensity", "z", "i"], False))

        def test_Jtvec_adjointTest_hxr_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["MagneticField", "x", "r"], False))

        def test_Jtvec_adjointTest_hyr_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["MagneticField", "y", "r"], False))

        def test_Jtvec_adjointTest_hzr_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["MagneticField", "z", "r"], False))

        def test_Jtvec_adjointTest_hxi_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["MagneticField", "x", "i"], False))

        def test_Jtvec_adjointTest_hyi_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["MagneticField", "y", "i"], False))

        def test_Jtvec_adjointTest_hzi_Bform_FaceEdgeConductivity(self):
            self.assertTrue(adjointTest("b", ["MagneticField", "z", "i"], False))


if __name__ == "__main__":
    unittest.main()
