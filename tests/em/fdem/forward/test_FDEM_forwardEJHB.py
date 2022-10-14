import unittest
from scipy.constants import mu_0
from SimPEG.electromagnetics.utils.testing_utils import getFDEMProblem, crossCheckTest

testEJ = True
testBH = True

TOLEJHB = 1  # averaging and more sensitive to boundary condition violations (ie. the impact of violating the boundary conditions in each case is different.)
# TODO: choose better testing parameters to lower this

SrcList = ["RawVec", "MagDipole", "MagDipole_Bfield", "MagDipole", "CircularLoop"]


class FDEM_CrossCheck(unittest.TestCase):
    if testEJ:

        def test_EJ_CrossCheck_jxr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["CurrentDensity", "x", "r"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_jyr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["CurrentDensity", "y", "r"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_jzr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["CurrentDensity", "z", "r"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_jxi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["CurrentDensity", "x", "i"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_jyi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["CurrentDensity", "y", "i"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_jzi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["CurrentDensity", "z", "i"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_exr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["ElectricField", "x", "r"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_eyr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["ElectricField", "y", "r"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_ezr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["ElectricField", "z", "r"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_exi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["ElectricField", "x", "i"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_eyi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["ElectricField", "y", "i"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_ezi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["ElectricField", "z", "i"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_bxr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["MagneticFluxDensity", "x", "r"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_byr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["MagneticFluxDensity", "y", "r"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_bzr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["MagneticFluxDensity", "z", "r"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_bxi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["MagneticFluxDensity", "x", "i"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_byi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["MagneticFluxDensity", "y", "i"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_bzi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["MagneticFluxDensity", "z", "i"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_hxr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["MagneticField", "x", "r"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_hyr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["MagneticField", "y", "r"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_hzr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["MagneticField", "z", "r"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_hxi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["MagneticField", "x", "i"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_hyi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["MagneticField", "y", "i"], TOL=TOLEJHB
                )
            )

        def test_EJ_CrossCheck_hzi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "e", "j", ["MagneticField", "z", "i"], TOL=TOLEJHB
                )
            )

    if testBH:

        def test_HB_CrossCheck_jxr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["CurrentDensity", "x", "r"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_jyr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["CurrentDensity", "y", "r"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_jzr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["CurrentDensity", "z", "r"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_jxi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["CurrentDensity", "x", "i"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_jyi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["CurrentDensity", "y", "i"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_jzi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["CurrentDensity", "z", "i"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_exr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["ElectricField", "x", "r"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_eyr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["ElectricField", "y", "r"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_ezr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["ElectricField", "z", "r"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_exi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["ElectricField", "x", "i"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_eyi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["ElectricField", "y", "i"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_ezi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["ElectricField", "z", "i"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_bxr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["MagneticFluxDensity", "x", "r"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_byr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["MagneticFluxDensity", "y", "r"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_bzr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["MagneticFluxDensity", "z", "r"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_bxi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["MagneticFluxDensity", "x", "i"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_byi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["MagneticFluxDensity", "y", "i"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_bzi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["MagneticFluxDensity", "z", "i"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_hxr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["MagneticField", "x", "r"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_hyr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["MagneticField", "y", "r"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_hzr_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["MagneticField", "z", "r"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_hxi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["MagneticField", "x", "i"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_hyi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["MagneticField", "y", "i"], TOL=TOLEJHB
                )
            )

        def test_HB_CrossCheck_hzi_Jform(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList, "h", "b", ["MagneticField", "z", "i"], TOL=TOLEJHB
                )
            )