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

TOLEBHJ = 1e-5
TOLEJHB = 1 # averaging and more sensitive to boundary condition violations (ie. the impact of violating the boundary conditions in each case is different.)
#TODO: choose better testing parameters to lower this 

SrcList = ['RawVec', 'MagDipole_Bfield', 'MagDipole', 'CircularLoop']


class FDEM_CrossCheck(unittest.TestCase):
    if testEJ:
        def test_EJ_CrossCheck_jxr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'jxr', TOL=TOLEJHB))
        def test_EJ_CrossCheck_jyr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'jyr',  TOL=TOLEJHB))
        def test_EJ_CrossCheck_jzr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'jzr',  TOL=TOLEJHB))
        def test_EJ_CrossCheck_jxi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'jxi',  TOL=TOLEJHB))
        def test_EJ_CrossCheck_jyi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'jyi',  TOL=TOLEJHB))
        def test_EJ_CrossCheck_jzi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'jzi',  TOL=TOLEJHB))

        def test_EJ_CrossCheck_exr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'exr',  TOL=TOLEJHB))
        def test_EJ_CrossCheck_eyr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'eyr',  TOL=TOLEJHB))
        def test_EJ_CrossCheck_ezr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'ezr',  TOL=TOLEJHB))
        def test_EJ_CrossCheck_exi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'exi',  TOL=TOLEJHB))
        def test_EJ_CrossCheck_eyi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'eyi',  TOL=TOLEJHB))
        def test_EJ_CrossCheck_ezi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'ezi',  TOL=TOLEJHB))

        def test_EJ_CrossCheck_bxr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'bxr',  TOL=TOLEJHB))
        def test_EJ_CrossCheck_byr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'byr',  TOL=TOLEJHB))
        def test_EJ_CrossCheck_bzr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'bzr',  TOL=TOLEJHB))
        def test_EJ_CrossCheck_bxi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'bxi',  TOL=TOLEJHB))
        def test_EJ_CrossCheck_byi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'byi',  TOL=TOLEJHB))
        def test_EJ_CrossCheck_bzi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'bzi',  TOL=TOLEJHB))

        def test_EJ_CrossCheck_hxr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'hxr',  TOL=TOLEJHB))
        def test_EJ_CrossCheck_hyr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'hyr',  TOL=TOLEJHB))
        def test_EJ_CrossCheck_hzr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'hzr',  TOL=TOLEJHB))
        def test_EJ_CrossCheck_hxi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'hxi',  TOL=TOLEJHB))
        def test_EJ_CrossCheck_hyi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'hyi',  TOL=TOLEJHB))
        def test_EJ_CrossCheck_hzi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'j', 'hzi',  TOL=TOLEJHB))

if __name__ == '__main__':
    unittest.main()