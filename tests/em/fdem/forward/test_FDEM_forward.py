import unittest
import numpy as np
import sys
from scipy.constants import mu_0

from SimPEG import EM
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


class SrcLocTest(unittest.TestCase):
    def test_src(self):
        src = EM.FDEM.Src.MagDipole(
            [], loc=np.array([[1.5, 3., 5.]]),
            freq=10
        )
        self.assertTrue(np.all(src.loc == np.r_[1.5, 3., 5.]))
        self.assertTrue(src.loc.shape==(3,))

        with self.assertRaises(Exception):
            src = EM.FDEM.Src.MagDipole(
                [], loc=np.array([[0., 0., 0., 1.]]),
                freq=10
            )

        with self.assertRaises(Exception):
            src = EM.FDEM.Src.MagDipole(
                [], loc=np.r_[0., 0., 0., 1.],
                freq=10
            )

        src = EM.TDEM.Src.MagDipole(
            [], loc=np.array([[1.5, 3., 5.]]),
        )
        self.assertTrue(np.all(src.loc == np.r_[1.5, 3., 5.]))

        with self.assertRaises(Exception):
            src = EM.TDEM.Src.MagDipole(
                [], loc=np.array([[0., 0., 0., 1.]]),
            )

        with self.assertRaises(Exception):
            src = EM.TDEM.Src.MagDipole(
                [], loc=np.r_[0., 0., 0., 1.],
            )

class FDEM_CrossCheck(unittest.TestCase):
    if testEB:
        def test_EB_CrossCheck_exr_Eform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'b', 'exr', verbose=verbose))
        def test_EB_CrossCheck_eyr_Eform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'b', 'eyr', verbose=verbose))
        def test_EB_CrossCheck_ezr_Eform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'b', 'ezr', verbose=verbose))
        def test_EB_CrossCheck_exi_Eform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'b', 'exi', verbose=verbose))
        def test_EB_CrossCheck_eyi_Eform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'b', 'eyi', verbose=verbose))
        def test_EB_CrossCheck_ezi_Eform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'b', 'ezi', verbose=verbose))

        def test_EB_CrossCheck_bxr_Eform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'b', 'bxr', verbose=verbose))
        def test_EB_CrossCheck_byr_Eform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'b', 'byr', verbose=verbose))
        def test_EB_CrossCheck_bzr_Eform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'b', 'bzr', verbose=verbose))
        def test_EB_CrossCheck_bxi_Eform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'b', 'bxi', verbose=verbose))
        def test_EB_CrossCheck_byi_Eform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'b', 'byi', verbose=verbose))
        def test_EB_CrossCheck_bzi_Eform(self):
            self.assertTrue(crossCheckTest(SrcList, 'e', 'b', 'bzi', verbose=verbose))

    if testHJ:
        def test_HJ_CrossCheck_jxr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'j', 'h', 'jxr', verbose=verbose))
        def test_HJ_CrossCheck_jyr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'j', 'h', 'jyr', verbose=verbose))
        def test_HJ_CrossCheck_jzr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'j', 'h', 'jzr', verbose=verbose))
        def test_HJ_CrossCheck_jxi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'j', 'h', 'jxi', verbose=verbose))
        def test_HJ_CrossCheck_jyi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'j', 'h', 'jyi', verbose=verbose))
        def test_HJ_CrossCheck_jzi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'j', 'h', 'jzi', verbose=verbose))

        def test_HJ_CrossCheck_hxr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'j', 'h', 'hxr', verbose=verbose))
        def test_HJ_CrossCheck_hyr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'j', 'h', 'hyr', verbose=verbose))
        def test_HJ_CrossCheck_hzr_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'j', 'h', 'hzr', verbose=verbose))
        def test_HJ_CrossCheck_hxi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'j', 'h', 'hxi', verbose=verbose))
        def test_HJ_CrossCheck_hyi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'j', 'h', 'hyi', verbose=verbose))
        def test_HJ_CrossCheck_hzi_Jform(self):
            self.assertTrue(crossCheckTest(SrcList, 'j', 'h', 'hzi', verbose=verbose))

if __name__ == '__main__':
    unittest.main()
