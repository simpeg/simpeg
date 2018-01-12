from __future__ import print_function
import unittest
import numpy as np
from SimPEG import EM
from scipy.constants import mu_0
from SimPEG.EM.Utils.testingUtils import getFDEMProblem

testE = True
testB = True

verbose = False

TOL = 1e-5
FLR = 1e-20 # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = 1e-1
addrandoms = True

SrcList = ['RawVec', 'MagDipole'] #or 'MAgDipole_Bfield', 'CircularLoop', 'RawVec'


def adjointTest(fdemType, comp):
    prb = getFDEMProblem(fdemType, comp, SrcList, freq)
    # prb.solverOpts = dict(check_accuracy=True)
    print('Adjoint {0!s} formulation - {1!s}'.format(fdemType, comp))

    m  = np.log(np.ones(prb.sigmaMap.nP)*CONDUCTIVITY)
    mu = np.ones(prb.mesh.nC)*MU

    if addrandoms is True:
        m  = m + np.random.randn(prb.sigmaMap.nP)*np.log(CONDUCTIVITY)*1e-1
        mu = mu + np.random.randn(prb.mesh.nC)*MU*1e-1

    survey = prb.survey
    # prb.PropMap.PropModel.mu = mu
    # prb.PropMap.PropModel.mui = 1./mu
    u = prb.fields(m)

    v = np.random.rand(survey.nD)
    w = np.random.rand(prb.mesh.nC)

    vJw = v.dot(prb.Jvec(m, w, u))
    wJtv = w.dot(prb.Jtvec(m, v, u))
    tol = np.max([TOL*(10**int(np.log10(np.abs(vJw)))),FLR])
    print(vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol)
    return np.abs(vJw - wJtv) < tol

class FDEM_AdjointTests(unittest.TestCase):
    if testE:
        def test_Jtvec_adjointTest_exr_Eform(self):
            self.assertTrue(adjointTest('e', 'exr'))
        def test_Jtvec_adjointTest_eyr_Eform(self):
            self.assertTrue(adjointTest('e', 'eyr'))
        def test_Jtvec_adjointTest_ezr_Eform(self):
            self.assertTrue(adjointTest('e', 'ezr'))
        def test_Jtvec_adjointTest_exi_Eform(self):
            self.assertTrue(adjointTest('e', 'exi'))
        def test_Jtvec_adjointTest_eyi_Eform(self):
            self.assertTrue(adjointTest('e', 'eyi'))
        def test_Jtvec_adjointTest_ezi_Eform(self):
            self.assertTrue(adjointTest('e', 'ezi'))

        def test_Jtvec_adjointTest_bxr_Eform(self):
            self.assertTrue(adjointTest('e', 'bxr'))
        def test_Jtvec_adjointTest_byr_Eform(self):
            self.assertTrue(adjointTest('e', 'byr'))
        def test_Jtvec_adjointTest_bzr_Eform(self):
            self.assertTrue(adjointTest('e', 'bzr'))
        def test_Jtvec_adjointTest_bxi_Eform(self):
            self.assertTrue(adjointTest('e', 'bxi'))
        def test_Jtvec_adjointTest_byi_Eform(self):
            self.assertTrue(adjointTest('e', 'byi'))
        def test_Jtvec_adjointTest_bzi_Eform(self):
            self.assertTrue(adjointTest('e', 'bzi'))

        def test_Jtvec_adjointTest_jxr_Eform(self):
            self.assertTrue(adjointTest('e', 'jxr'))
        def test_Jtvec_adjointTest_jyr_Eform(self):
            self.assertTrue(adjointTest('e', 'jyr'))
        def test_Jtvec_adjointTest_jzr_Eform(self):
            self.assertTrue(adjointTest('e', 'jzr'))
        def test_Jtvec_adjointTest_jxi_Eform(self):
            self.assertTrue(adjointTest('e', 'jxi'))
        def test_Jtvec_adjointTest_jyi_Eform(self):
            self.assertTrue(adjointTest('e', 'jyi'))
        def test_Jtvec_adjointTest_jzi_Eform(self):
            self.assertTrue(adjointTest('e', 'jzi'))

        def test_Jtvec_adjointTest_hxr_Eform(self):
            self.assertTrue(adjointTest('e', 'hxr'))
        def test_Jtvec_adjointTest_hyr_Eform(self):
            self.assertTrue(adjointTest('e', 'hyr'))
        def test_Jtvec_adjointTest_hzr_Eform(self):
            self.assertTrue(adjointTest('e', 'hzr'))
        def test_Jtvec_adjointTest_hxi_Eform(self):
            self.assertTrue(adjointTest('e', 'hxi'))
        def test_Jtvec_adjointTest_hyi_Eform(self):
            self.assertTrue(adjointTest('e', 'hyi'))
        def test_Jtvec_adjointTest_hzi_Eform(self):
            self.assertTrue(adjointTest('e', 'hzi'))

    if testB:
        def test_Jtvec_adjointTest_exr_Bform(self):
            self.assertTrue(adjointTest('b', 'exr'))
        def test_Jtvec_adjointTest_eyr_Bform(self):
            self.assertTrue(adjointTest('b', 'eyr'))
        def test_Jtvec_adjointTest_ezr_Bform(self):
            self.assertTrue(adjointTest('b', 'ezr'))
        def test_Jtvec_adjointTest_exi_Bform(self):
            self.assertTrue(adjointTest('b', 'exi'))
        def test_Jtvec_adjointTest_eyi_Bform(self):
            self.assertTrue(adjointTest('b', 'eyi'))
        def test_Jtvec_adjointTest_ezi_Bform(self):
            self.assertTrue(adjointTest('b', 'ezi'))

        def test_Jtvec_adjointTest_bxr_Bform(self):
            self.assertTrue(adjointTest('b', 'bxr'))
        def test_Jtvec_adjointTest_byr_Bform(self):
            self.assertTrue(adjointTest('b', 'byr'))
        def test_Jtvec_adjointTest_bzr_Bform(self):
            self.assertTrue(adjointTest('b', 'bzr'))
        def test_Jtvec_adjointTest_bxi_Bform(self):
            self.assertTrue(adjointTest('b', 'bxi'))
        def test_Jtvec_adjointTest_byi_Bform(self):
            self.assertTrue(adjointTest('b', 'byi'))
        def test_Jtvec_adjointTest_bzi_Bform(self):
            self.assertTrue(adjointTest('b', 'bzi'))

        def test_Jtvec_adjointTest_jxr_Bform(self):
            self.assertTrue(adjointTest('b', 'jxr'))
        def test_Jtvec_adjointTest_jyr_Bform(self):
            self.assertTrue(adjointTest('b', 'jyr'))
        def test_Jtvec_adjointTest_jzr_Bform(self):
            self.assertTrue(adjointTest('b', 'jzr'))
        def test_Jtvec_adjointTest_jxi_Bform(self):
            self.assertTrue(adjointTest('b', 'jxi'))
        def test_Jtvec_adjointTest_jyi_Bform(self):
            self.assertTrue(adjointTest('b', 'jyi'))
        def test_Jtvec_adjointTest_jzi_Bform(self):
            self.assertTrue(adjointTest('b', 'jzi'))

        def test_Jtvec_adjointTest_hxr_Bform(self):
            self.assertTrue(adjointTest('b', 'hxr'))
        def test_Jtvec_adjointTest_hyr_Bform(self):
            self.assertTrue(adjointTest('b', 'hyr'))
        def test_Jtvec_adjointTest_hzr_Bform(self):
            self.assertTrue(adjointTest('b', 'hzr'))
        def test_Jtvec_adjointTest_hxi_Bform(self):
            self.assertTrue(adjointTest('b', 'hxi'))
        def test_Jtvec_adjointTest_hyi_Bform(self):
            self.assertTrue(adjointTest('b', 'hyi'))
        def test_Jtvec_adjointTest_hzi_Bform(self):
            self.assertTrue(adjointTest('b', 'hzi'))


if __name__ == '__main__':
    unittest.main()
