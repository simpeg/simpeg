import unittest
from SimPEG import *
from SimPEG import EM
import sys
from scipy.constants import mu_0
from SimPEG.EM.Utils.testingUtils import getFDEMProblem

testDerivs = True
testEB = True
testHJ = True

verbose = False

TOL = 1e-5
FLR = 1e-20 # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = 1e-1
addrandoms = True

SrcType = 'RawVec' #or 'MAgDipole_Bfield', 'CircularLoop', 'RawVec'


def derivTest(fdemType, comp):

    prb = getFDEMProblem(fdemType, comp, [SrcType], freq)
    print '%s formulation - %s' % (fdemType, comp)
    x0 = np.log(np.ones(prb.mapping.nP)*CONDUCTIVITY)
    mu = np.log(np.ones(prb.mesh.nC)*MU)

    if addrandoms is True:
        x0 = x0 + np.random.randn(prb.mapping.nP)*np.log(CONDUCTIVITY)*1e-1
        mu = mu + np.random.randn(prb.mapping.nP)*MU*1e-1

    # prb.PropMap.PropModel.mu = mu
    # prb.PropMap.PropModel.mui = 1./mu

    survey = prb.survey
    def fun(x):
        return survey.dpred(x), lambda x: prb.Jvec(x0, x)
    return Tests.checkDerivative(fun, x0, num=2, plotIt=False, eps=FLR)


class FDEM_DerivTests(unittest.TestCase):

    if testEB:
        def test_Jvec_exr_Eform(self):
            self.assertTrue(derivTest('e', 'exr'))
        def test_Jvec_eyr_Eform(self):
            self.assertTrue(derivTest('e', 'eyr'))
        def test_Jvec_ezr_Eform(self):
            self.assertTrue(derivTest('e', 'ezr'))
        def test_Jvec_exi_Eform(self):
            self.assertTrue(derivTest('e', 'exi'))
        def test_Jvec_eyi_Eform(self):
            self.assertTrue(derivTest('e', 'eyi'))
        def test_Jvec_ezi_Eform(self):
            self.assertTrue(derivTest('e', 'ezi'))

        def test_Jvec_bxr_Eform(self):
            self.assertTrue(derivTest('e', 'bxr'))
        def test_Jvec_byr_Eform(self):
            self.assertTrue(derivTest('e', 'byr'))
        def test_Jvec_bzr_Eform(self):
            self.assertTrue(derivTest('e', 'bzr'))
        def test_Jvec_bxi_Eform(self):
            self.assertTrue(derivTest('e', 'bxi'))
        def test_Jvec_byi_Eform(self):
            self.assertTrue(derivTest('e', 'byi'))
        def test_Jvec_bzi_Eform(self):
            self.assertTrue(derivTest('e', 'bzi'))

        def test_Jvec_exr_Bform(self):
            self.assertTrue(derivTest('b', 'exr'))
        def test_Jvec_eyr_Bform(self):
            self.assertTrue(derivTest('b', 'eyr'))
        def test_Jvec_ezr_Bform(self):
            self.assertTrue(derivTest('b', 'ezr'))
        def test_Jvec_exi_Bform(self):
            self.assertTrue(derivTest('b', 'exi'))
        def test_Jvec_eyi_Bform(self):
            self.assertTrue(derivTest('b', 'eyi'))
        def test_Jvec_ezi_Bform(self):
            self.assertTrue(derivTest('b', 'ezi'))

        def test_Jvec_bxr_Bform(self):
            self.assertTrue(derivTest('b', 'bxr'))
        def test_Jvec_byr_Bform(self):
            self.assertTrue(derivTest('b', 'byr'))
        def test_Jvec_bzr_Bform(self):
            self.assertTrue(derivTest('b', 'bzr'))
        def test_Jvec_bxi_Bform(self):
            self.assertTrue(derivTest('b', 'bxi'))
        def test_Jvec_byi_Bform(self):
            self.assertTrue(derivTest('b', 'byi'))
        def test_Jvec_bzi_Bform(self):
            self.assertTrue(derivTest('b', 'bzi'))

    if testHJ:
        def test_Jvec_jxr_Jform(self):
            self.assertTrue(derivTest('j', 'jxr'))
        def test_Jvec_jyr_Jform(self):
            self.assertTrue(derivTest('j', 'jyr'))
        def test_Jvec_jzr_Jform(self):
            self.assertTrue(derivTest('j', 'jzr'))
        def test_Jvec_jxi_Jform(self):
            self.assertTrue(derivTest('j', 'jxi'))
        def test_Jvec_jyi_Jform(self):
            self.assertTrue(derivTest('j', 'jyi'))
        def test_Jvec_jzi_Jform(self):
            self.assertTrue(derivTest('j', 'jzi'))

        def test_Jvec_hxr_Jform(self):
            self.assertTrue(derivTest('j', 'hxr'))
        def test_Jvec_hyr_Jform(self):
            self.assertTrue(derivTest('j', 'hyr'))
        def test_Jvec_hzr_Jform(self):
            self.assertTrue(derivTest('j', 'hzr'))
        def test_Jvec_hxi_Jform(self):
            self.assertTrue(derivTest('j', 'hxi'))
        def test_Jvec_hyi_Jform(self):
            self.assertTrue(derivTest('j', 'hyi'))
        def test_Jvec_hzi_Jform(self):
            self.assertTrue(derivTest('j', 'hzi'))

        def test_Jvec_hxr_Hform(self):
            self.assertTrue(derivTest('h', 'hxr'))
        def test_Jvec_hyr_Hform(self):
            self.assertTrue(derivTest('h', 'hyr'))
        def test_Jvec_hzr_Hform(self):
            self.assertTrue(derivTest('h', 'hzr'))
        def test_Jvec_hxi_Hform(self):
            self.assertTrue(derivTest('h', 'hxi'))
        def test_Jvec_hyi_Hform(self):
            self.assertTrue(derivTest('h', 'hyi'))
        def test_Jvec_hzi_Hform(self):
            self.assertTrue(derivTest('h', 'hzi'))

        def test_Jvec_hxr_Hform(self):
            self.assertTrue(derivTest('h', 'jxr'))
        def test_Jvec_hyr_Hform(self):
            self.assertTrue(derivTest('h', 'jyr'))
        def test_Jvec_hzr_Hform(self):
            self.assertTrue(derivTest('h', 'jzr'))
        def test_Jvec_hxi_Hform(self):
            self.assertTrue(derivTest('h', 'jxi'))
        def test_Jvec_hyi_Hform(self):
            self.assertTrue(derivTest('h', 'jyi'))
        def test_Jvec_hzi_Hform(self):
            self.assertTrue(derivTest('h', 'jzi'))


if __name__ == '__main__':
    unittest.main()
