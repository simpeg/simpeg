import unittest
from SimPEG import EM
from scipy.constants import mu_0
from SimPEG.EM.Utils.NewTestingUtils import getFDEMSimulation, crossCheckTest

testEB = True
testHJ = True
testEJ = True
testBH = True
verbose = False

TOLEJHB = 1 # averaging and more sensitive to boundary condition violations (ie. the impact of violating the boundary conditions in each case is different.)
#TODO: choose better testing parameters to lower this

SrcList = ['RawVec', 'MagDipole_Bfield', 'MagDipole', 'CircularLoop']


class FDEM_CrossCheck(unittest.TestCase):
    if testBH:
        def test_BH_CrossCheck_jxr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'jxr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_jyr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'jyr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_jzr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'jzr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_jxi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'jxi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_jyi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'jyi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_jzi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'jzi', verbose=verbose, TOL=TOLEJHB))

        def test_BH_CrossCheck_exr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'exr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_eyr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'eyr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_ezr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'ezr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_exi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'exi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_eyi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'eyi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_ezi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'ezi', verbose=verbose, TOL=TOLEJHB))

        def test_BH_CrossCheck_bxr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'bxr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_byr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'byr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_bzr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'bzr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_bxi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'bxi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_byi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'byi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_bzi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'bzi', verbose=verbose, TOL=TOLEJHB))

        def test_BH_CrossCheck_hxr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'hxr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_hyr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'hyr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_hzr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'hzr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_hxi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'hxi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_hyi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'hyi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_hzi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'hzi', verbose=verbose, TOL=TOLEJHB))

    if testBH:
        def test_BH_CrossCheck_jxr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'jxr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_jyr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'jyr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_jzr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'jzr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_jxi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'jxi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_jyi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'jyi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_jzi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'jzi', verbose=verbose, TOL=TOLEJHB))

        def test_BH_CrossCheck_exr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'exr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_eyr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'eyr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_ezr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'ezr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_exi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'exi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_eyi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'eyi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_ezi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'ezi', verbose=verbose, TOL=TOLEJHB))

        def test_BH_CrossCheck_bxr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'bxr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_byr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'byr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_bzr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'bzr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_bxi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'bxi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_byi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'byi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_bzi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'bzi', verbose=verbose, TOL=TOLEJHB))

        def test_BH_CrossCheck_hxr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'hxr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_hyr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'hyr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_hzr(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'hzr', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_hxi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'hxi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_hyi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'hyi', verbose=verbose, TOL=TOLEJHB))
        def test_BH_CrossCheck_hzi(self):
            self.assertTrue(crossCheckTest(SrcList, 'b', 'h', 'hzi', verbose=verbose, TOL=TOLEJHB))

if __name__ == '__main__':
    unittest.main()
