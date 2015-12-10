import unittest
from SimPEG import *
from SimPEG import EM
import sys
from scipy.constants import mu_0
from SimPEG.EM.Utils.testingUtils import getFDEMProblem, crossCheckTest

testEB = True
testHJ = True
testEJ = True
testBH = True
verbose = False

TOLEBHJ = 1e-5
TOLEJHB = 1 # averaging and more sensitive to boundary condition violations (ie. the impact of violating the boundary conditions in each case is different.)
#TODO: choose better testing parameters to lower this 

SrcList = ['RawVec', 'MagDipole_Bfield', 'MagDipole', 'CircularLoop']


class FDEM_CrossCheck(unittest.TestCase):
    if testEJ:
        def test_EJ_CrossCheck_jxr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'jxr', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_jyr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'jyr', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_jzr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'jzr', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_jxi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'jxi', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_jyi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'jyi', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_jzi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'jzi', verbose=verbose, TOL=TOLEJHB))

        def test_EJ_CrossCheck_exr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'exr', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_eyr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'eyr', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_ezr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'ezr', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_exi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'exi', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_eyi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'eyi', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_ezi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'ezi', verbose=verbose, TOL=TOLEJHB))

        def test_EJ_CrossCheck_bxr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'bxr', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_byr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'byr', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_bzr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'bzr', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_bxi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'bxi', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_byi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'byi', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_bzi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'bzi', verbose=verbose, TOL=TOLEJHB))

        def test_EJ_CrossCheck_hxr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'hxr', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_hyr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'hyr', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_hzr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'hzr', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_hxi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'hxi', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_hyi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'hyi', verbose=verbose, TOL=TOLEJHB))
        def test_EJ_CrossCheck_hzi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'hzi', verbose=verbose, TOL=TOLEJHB))

if __name__ == '__main__':
    unittest.main()