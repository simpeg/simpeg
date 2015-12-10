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
    if testBH:
        def test_BH_CrossCheck_jxr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'jxr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_jyr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'jyr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_jzr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'jzr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_jxi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'jxi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_jyi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'jyi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_jzi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'jzi', verbose=verbose, TOL=TOLEJHB))

        def test_BH_CrossCheck_exr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'exr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_eyr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'eyr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_ezr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'ezr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_exi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'exi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_eyi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'eyi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_ezi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'ezi', verbose=verbose, TOL=TOLEJHB))

        def test_BH_CrossCheck_bxr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'bxr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_byr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'byr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_bzr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'bzr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_bxi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'bxi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_byi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'byi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_bzi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'bzi', verbose=verbose, TOL=TOLEJHB))

        def test_BH_CrossCheck_hxr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'hxr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_hyr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'hyr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_hzr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'hzr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_hxi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'hxi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_hyi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'hyi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_hzi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'hzi', verbose=verbose, TOL=TOLEJHB))

if __name__ == '__main__':
    unittest.main()