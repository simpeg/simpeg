import unittest
from SimPEG import *
from SimPEG import EM
import sys
from scipy.constants import mu_0
from SimPEG.EM.Utils.testingUtils import getFDEMProblem

testEB = True
testHJ = True
testEJ = False
verbose = False

TOLEBHJ = 1e-5
TOLEJHB = 1e-1

FLR = 1e-20 # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = 1e-1
addrandoms = False

SrcList = ['RawVec', 'MagDipole_Bfield', 'MagDipole', 'CircularLoop']


def crossCheckTest(fdemType1, fdemType2, comp, TOL=TOLEBHJ):

    l2norm = lambda r: np.sqrt(r.dot(r))

    prb1 = getFDEMProblem(fdemType1, comp, SrcList, freq, verbose)
    mesh = prb1.mesh
    print 'Cross Checking Forward: %s formulation - %s' % (fdemType1, comp)
    m = np.log(np.ones(mesh.nC)*CONDUCTIVITY)
    mu = np.log(np.ones(mesh.nC)*MU)

    if addrandoms is True:
        m  = m + np.random.randn(mesh.nC)*np.log(CONDUCTIVITY)*1e-1
        mu = mu + np.random.randn(mesh.nC)*MU*1e-1

    # prb1.PropMap.PropModel.mu = mu
    # prb1.PropMap.PropModel.mui = 1./mu
    survey1 = prb1.survey
    d1 = survey1.dpred(m)

    if verbose:
        print '  Problem 1 solved'


    prb2 = getFDEMProblem(fdemType2, comp, SrcList, freq, verbose)

    # prb2.mu = mu
    survey2 = prb2.survey
    d2 = survey2.dpred(m)

    if verbose:
        print '  Problem 2 solved'

    r = d2-d1
    l2r = l2norm(r)

    tol = np.max([TOL*(10**int(np.log10(l2norm(d1)))),FLR])
    print l2norm(d1), l2norm(d2),  l2r , tol, l2r < tol
    return l2r < tol


class FDEM_CrossCheck(unittest.TestCase):
    if testEB:
        def test_EB_CrossCheck_exr_Eform(self):
            self.assertTrue(crossCheckTest('e', 'b', 'exr'))
        def test_EB_CrossCheck_eyr_Eform(self):
            self.assertTrue(crossCheckTest('e', 'b', 'eyr'))
        def test_EB_CrossCheck_ezr_Eform(self):
            self.assertTrue(crossCheckTest('e', 'b', 'ezr'))
        def test_EB_CrossCheck_exi_Eform(self):
            self.assertTrue(crossCheckTest('e', 'b', 'exi'))
        def test_EB_CrossCheck_eyi_Eform(self):
            self.assertTrue(crossCheckTest('e', 'b', 'eyi'))
        def test_EB_CrossCheck_ezi_Eform(self):
            self.assertTrue(crossCheckTest('e', 'b', 'ezi'))

        def test_EB_CrossCheck_bxr_Eform(self):
            self.assertTrue(crossCheckTest('e', 'b', 'bxr'))
        def test_EB_CrossCheck_byr_Eform(self):
            self.assertTrue(crossCheckTest('e', 'b', 'byr'))
        def test_EB_CrossCheck_bzr_Eform(self):
            self.assertTrue(crossCheckTest('e', 'b', 'bzr'))
        def test_EB_CrossCheck_bxi_Eform(self):
            self.assertTrue(crossCheckTest('e', 'b', 'bxi'))
        def test_EB_CrossCheck_byi_Eform(self):
            self.assertTrue(crossCheckTest('e', 'b', 'byi'))
        def test_EB_CrossCheck_bzi_Eform(self):
            self.assertTrue(crossCheckTest('e', 'b', 'bzi'))

    if testHJ:
        def test_HJ_CrossCheck_jxr_Jform(self):
            self.assertTrue(crossCheckTest('j', 'h', 'jxr'))
        def test_HJ_CrossCheck_jyr_Jform(self):
            self.assertTrue(crossCheckTest('j', 'h', 'jyr'))
        def test_HJ_CrossCheck_jzr_Jform(self):
            self.assertTrue(crossCheckTest('j', 'h', 'jzr'))
        def test_HJ_CrossCheck_jxi_Jform(self):
            self.assertTrue(crossCheckTest('j', 'h', 'jxi'))
        def test_HJ_CrossCheck_jyi_Jform(self):
            self.assertTrue(crossCheckTest('j', 'h', 'jyi'))
        def test_HJ_CrossCheck_jzi_Jform(self):
            self.assertTrue(crossCheckTest('j', 'h', 'jzi'))

        def test_HJ_CrossCheck_hxr_Jform(self):
            self.assertTrue(crossCheckTest('j', 'h', 'hxr'))
        def test_HJ_CrossCheck_hyr_Jform(self):
            self.assertTrue(crossCheckTest('j', 'h', 'hyr'))
        def test_HJ_CrossCheck_hzr_Jform(self):
            self.assertTrue(crossCheckTest('j', 'h', 'hzr'))
        def test_HJ_CrossCheck_hxi_Jform(self):
            self.assertTrue(crossCheckTest('j', 'h', 'hxi'))
        def test_HJ_CrossCheck_hyi_Jform(self):
            self.assertTrue(crossCheckTest('j', 'h', 'hyi'))
        def test_HJ_CrossCheck_hzi_Jform(self):
            self.assertTrue(crossCheckTest('j', 'h', 'hzi'))

    if testEJ:
        # def test_EJ_CrossCheck_jxr_Jform(self):
        #     self.assertTrue(crossCheckTest('e', 'j', 'jxr'))
        # def test_EJ_CrossCheck_jyr_Jform(self):
        #     self.assertTrue(crossCheckTest('e', 'j', 'jyr'))
        # def test_EJ_CrossCheck_jzr_Jform(self):
        #     self.assertTrue(crossCheckTest('e', 'j', 'jzr'))
        # def test_EJ_CrossCheck_jxi_Jform(self):
        #     self.assertTrue(crossCheckTest('e', 'j', 'jxi'))
        # def test_EJ_CrossCheck_jyi_Jform(self):
        #     self.assertTrue(crossCheckTest('e', 'j', 'jyi'))
        # def test_EJ_CrossCheck_jzi_Jform(self):
        #     self.assertTrue(crossCheckTest('e', 'j', 'jzi'))

        def test_EJ_CrossCheck_jxr_Jform(self):
            self.assertTrue(crossCheckTest('e', 'j', 'exr', TOL=TOLEJHB))
        # def test_EJ_CrossCheck_jyr_Jform(self):
        #     self.assertTrue(crossCheckTest('e', 'j', 'eyr'))
        # def test_EJ_CrossCheck_jzr_Jform(self):
        #     self.assertTrue(crossCheckTest('e', 'j', 'ezr'))
        # def test_EJ_CrossCheck_jxi_Jform(self):
        #     self.assertTrue(crossCheckTest('e', 'j', 'exi'))
        # def test_EJ_CrossCheck_jyi_Jform(self):
        #     self.assertTrue(crossCheckTest('e', 'j', 'eyi'))
        # def test_EJ_CrossCheck_jzi_Jform(self):
        #     self.assertTrue(crossCheckTest('e', 'j', 'ezi'))

if __name__ == '__main__':
    unittest.main()