from __future__ import print_function
import unittest
import numpy as np
from SimPEG import EM
from scipy.constants import mu_0
from SimPEG.EM.Utils.testingUtils import getFDEMProblem

testJ = True
testH = True

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
    u = prb.fields(m)

    v = np.random.rand(survey.nD)
    w = np.random.rand(prb.mesh.nC)

    vJw = v.dot(prb.Jvec(m, w, u))
    wJtv = w.dot(prb.Jtvec(m, v, u))
    tol = np.max([TOL*(10**int(np.log10(np.abs(vJw)))),FLR])
    print(vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol)
    return np.abs(vJw - wJtv) < tol

class FDEM_AdjointTests(unittest.TestCase):

    if testJ:
        def test_Jtvec_adjointTest_jxr_Jform(self):
            self.assertTrue(adjointTest('j', 'jxr'))
        def test_Jtvec_adjointTest_jyr_Jform(self):
            self.assertTrue(adjointTest('j', 'jyr'))
        def test_Jtvec_adjointTest_jzr_Jform(self):
            self.assertTrue(adjointTest('j', 'jzr'))
        def test_Jtvec_adjointTest_jxi_Jform(self):
            self.assertTrue(adjointTest('j', 'jxi'))
        def test_Jtvec_adjointTest_jyi_Jform(self):
            self.assertTrue(adjointTest('j', 'jyi'))
        def test_Jtvec_adjointTest_jzi_Jform(self):
            self.assertTrue(adjointTest('j', 'jzi'))

        def test_Jtvec_adjointTest_hxr_Jform(self):
            self.assertTrue(adjointTest('j', 'hxr'))
        def test_Jtvec_adjointTest_hyr_Jform(self):
            self.assertTrue(adjointTest('j', 'hyr'))
        def test_Jtvec_adjointTest_hzr_Jform(self):
            self.assertTrue(adjointTest('j', 'hzr'))
        def test_Jtvec_adjointTest_hxi_Jform(self):
            self.assertTrue(adjointTest('j', 'hxi'))
        def test_Jtvec_adjointTest_hyi_Jform(self):
            self.assertTrue(adjointTest('j', 'hyi'))
        def test_Jtvec_adjointTest_hzi_Jform(self):
            self.assertTrue(adjointTest('j', 'hzi'))

        def test_Jtvec_adjointTest_exr_Jform(self):
            self.assertTrue(adjointTest('j', 'exr'))
        def test_Jtvec_adjointTest_eyr_Jform(self):
            self.assertTrue(adjointTest('j', 'eyr'))
        def test_Jtvec_adjointTest_ezr_Jform(self):
            self.assertTrue(adjointTest('j', 'ezr'))
        def test_Jtvec_adjointTest_exi_Jform(self):
            self.assertTrue(adjointTest('j', 'exi'))
        def test_Jtvec_adjointTest_eyi_Jform(self):
            self.assertTrue(adjointTest('j', 'eyi'))
        def test_Jtvec_adjointTest_ezi_Jform(self):
            self.assertTrue(adjointTest('j', 'ezi'))

        def test_Jtvec_adjointTest_bxr_Jform(self):
            self.assertTrue(adjointTest('j', 'bxr'))
        def test_Jtvec_adjointTest_byr_Jform(self):
            self.assertTrue(adjointTest('j', 'byr'))
        def test_Jtvec_adjointTest_bzr_Jform(self):
            self.assertTrue(adjointTest('j', 'bzr'))
        def test_Jtvec_adjointTest_bxi_Jform(self):
            self.assertTrue(adjointTest('j', 'bxi'))
        def test_Jtvec_adjointTest_byi_Jform(self):
            self.assertTrue(adjointTest('j', 'byi'))
        def test_Jtvec_adjointTest_bzi_Jform(self):
            self.assertTrue(adjointTest('j', 'bzi'))

    if testH:
        def test_Jtvec_adjointTest_hxr_Hform(self):
            self.assertTrue(adjointTest('h', 'hxr'))
        def test_Jtvec_adjointTest_hyr_Hform(self):
            self.assertTrue(adjointTest('h', 'hyr'))
        def test_Jtvec_adjointTest_hzr_Hform(self):
            self.assertTrue(adjointTest('h', 'hzr'))
        def test_Jtvec_adjointTest_hxi_Hform(self):
            self.assertTrue(adjointTest('h', 'hxi'))
        def test_Jtvec_adjointTest_hyi_Hform(self):
            self.assertTrue(adjointTest('h', 'hyi'))
        def test_Jtvec_adjointTest_hzi_Hform(self):
            self.assertTrue(adjointTest('h', 'hzi'))

        def test_Jtvec_adjointTest_jxr_Hform(self):
            self.assertTrue(adjointTest('h', 'jxr'))
        def test_Jtvec_adjointTest_jyr_Hform(self):
            self.assertTrue(adjointTest('h', 'jyr'))
        def test_Jtvec_adjointTest_jzr_Hform(self):
            self.assertTrue(adjointTest('h', 'jzr'))
        def test_Jtvec_adjointTest_jxi_Hform(self):
            self.assertTrue(adjointTest('h', 'jxi'))
        def test_Jtvec_adjointTest_jyi_Hform(self):
            self.assertTrue(adjointTest('h', 'jyi'))
        def test_Jtvec_adjointTest_jzi_Hform(self):
            self.assertTrue(adjointTest('h', 'jzi'))

        def test_Jtvec_adjointTest_exr_Hform(self):
            self.assertTrue(adjointTest('h', 'exr'))
        def test_Jtvec_adjointTest_eyr_Hform(self):
            self.assertTrue(adjointTest('h', 'eyr'))
        def test_Jtvec_adjointTest_ezr_Hform(self):
            self.assertTrue(adjointTest('h', 'ezr'))
        def test_Jtvec_adjointTest_exi_Hform(self):
            self.assertTrue(adjointTest('h', 'exi'))
        def test_Jtvec_adjointTest_eyi_Hform(self):
            self.assertTrue(adjointTest('h', 'eyi'))
        def test_Jtvec_adjointTest_ezi_Hform(self):
            self.assertTrue(adjointTest('h', 'ezi'))

        def test_Jtvec_adjointTest_bxr_Hform(self):
            self.assertTrue(adjointTest('h', 'bxr'))
        def test_Jtvec_adjointTest_byr_Hform(self):
            self.assertTrue(adjointTest('h', 'byr'))
        def test_Jtvec_adjointTest_bzr_Hform(self):
            self.assertTrue(adjointTest('h', 'bzr'))
        def test_Jtvec_adjointTest_bxi_Hform(self):
            self.assertTrue(adjointTest('h', 'bxi'))
        def test_Jtvec_adjointTest_byi_Hform(self):
            self.assertTrue(adjointTest('h', 'byi'))
        def test_Jtvec_adjointTest_bzi_Hform(self):
            self.assertTrue(adjointTest('h', 'bzi'))


if __name__ == '__main__':
    unittest.main()
