import unittest
import numpy as np
import sys
from scipy.constants import mu_0

from SimPEG.electromagnetics import frequency_domain as fdem
from SimPEG.electromagnetics import time_domain as tdem
from SimPEG.electromagnetics.utils.testing_utils import getFDEMProblem, crossCheckTest

testEB = True
testHJ = True
testEJ = True
testBH = True
verbose = False

TOLEBHJ = 1e-5
TOLEJHB = 1  # averaging and more sensitive to boundary condition violations (ie. the impact of violating the boundary conditions in each case is different.)
# TODO: choose better testing parameters to lower this

SrcList = ["RawVec", "MagDipole_Bfield", "MagDipole", "CircularLoop"]


class SrcLocTest(unittest.TestCase):
    def test_src(self):
        src = fdem.Src.MagDipole([], loc=np.array([[1.5, 3.0, 5.0]]), freq=10)
        self.assertTrue(np.all(src.location == np.r_[1.5, 3.0, 5.0]))
        self.assertTrue(src.location.shape == (3,))

        with self.assertRaises(Exception):
            src = fdem.Src.MagDipole([], loc=np.array([[0.0, 0.0, 0.0, 1.0]]), freq=10)

        with self.assertRaises(Exception):
            src = fdem.Src.MagDipole([], loc=np.r_[0.0, 0.0, 0.0, 1.0], freq=10)

        src = tdem.Src.MagDipole([], loc=np.array([[1.5, 3.0, 5.0]]),)
        self.assertTrue(np.all(src.location == np.r_[1.5, 3.0, 5.0]))

        with self.assertRaises(Exception):
            src = tdem.Src.MagDipole([], loc=np.array([[0.0, 0.0, 0.0, 1.0]]),)

        with self.assertRaises(Exception):
            src = tdem.Src.MagDipole([], loc=np.r_[0.0, 0.0, 0.0, 1.0],)


class FDEM_CrossCheck(unittest.TestCase):
    if testEB:

        def test_EB_CrossCheck_exr_Eform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "b", ["ElectricField", "x", "r"], verbose=verbose
                )
            )

        def test_EB_CrossCheck_eyr_Eform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "b", ["ElectricField", "y", "r"], verbose=verbose
                )
            )

        def test_EB_CrossCheck_ezr_Eform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "b", ["ElectricField", "z", "r"], verbose=verbose
                )
            )

        def test_EB_CrossCheck_exi_Eform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "b", ["ElectricField", "x", "i"], verbose=verbose
                )
            )

        def test_EB_CrossCheck_eyi_Eform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "b", ["ElectricField", "y", "i"], verbose=verbose
                )
            )

        def test_EB_CrossCheck_ezi_Eform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "b", ["ElectricField", "z", "i"], verbose=verbose
                )
            )

        def test_EB_CrossCheck_bxr_Eform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "e",
                    "b",
                    ["MagneticFluxDensity", "x", "r"],
                    verbose=verbose,
                )
            )

        def test_EB_CrossCheck_byr_Eform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "e",
                    "b",
                    ["MagneticFluxDensity", "y", "r"],
                    verbose=verbose,
                )
            )

        def test_EB_CrossCheck_bzr_Eform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "e",
                    "b",
                    ["MagneticFluxDensity", "z", "r"],
                    verbose=verbose,
                )
            )

        def test_EB_CrossCheck_bxi_Eform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "e",
                    "b",
                    ["MagneticFluxDensity", "x", "i"],
                    verbose=verbose,
                )
            )

        def test_EB_CrossCheck_byi_Eform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "e",
                    "b",
                    ["MagneticFluxDensity", "y", "i"],
                    verbose=verbose,
                )
            )

        def test_EB_CrossCheck_bzi_Eform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "e",
                    "b",
                    ["MagneticFluxDensity", "z", "i"],
                    verbose=verbose,
                )
            )

    if testHJ:

        def test_HJ_CrossCheck_jxr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "j", "h", ["CurrentDensity", "x", "r"], verbose=verbose
                )
            )

        def test_HJ_CrossCheck_jyr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "j", "h", ["CurrentDensity", "y", "r"], verbose=verbose
                )
            )

        def test_HJ_CrossCheck_jzr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "j", "h", ["CurrentDensity", "z", "r"], verbose=verbose
                )
            )

        def test_HJ_CrossCheck_jxi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "j", "h", ["CurrentDensity", "x", "i"], verbose=verbose
                )
            )

        def test_HJ_CrossCheck_jyi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "j", "h", ["CurrentDensity", "y", "i"], verbose=verbose
                )
            )

        def test_HJ_CrossCheck_jzi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "j", "h", ["CurrentDensity", "z", "i"], verbose=verbose
                )
            )

        def test_HJ_CrossCheck_hxr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "j", "h", ["MagneticField", "x", "r"], verbose=verbose
                )
            )

        def test_HJ_CrossCheck_hyr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "j", "h", ["MagneticField", "y", "r"], verbose=verbose
                )
            )

        def test_HJ_CrossCheck_hzr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "j", "h", ["MagneticField", "z", "r"], verbose=verbose
                )
            )

        def test_HJ_CrossCheck_hxi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "j", "h", ["MagneticField", "x", "i"], verbose=verbose
                )
            )

        def test_HJ_CrossCheck_hyi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "j", "h", ["MagneticField", "y", "i"], verbose=verbose
                )
            )

        def test_HJ_CrossCheck_hzi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "j", "h", ["MagneticField", "z", "i"], verbose=verbose
                )
            )


if __name__ == "__main__":
    unittest.main()
