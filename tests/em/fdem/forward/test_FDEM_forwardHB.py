import unittest
from scipy.constants import mu_0
from SimPEG.electromagnetics.utils.testing_utils import getFDEMProblem, crossCheckTest

testEB = True
testHJ = True
testEJ = True
testBH = True
verbose = False

TOLEJHB = 1  # averaging and more sensitive to boundary condition violations (ie. the impact of violating the boundary conditions in each case is different.)
# TODO: choose better testing parameters to lower this

SrcList = ["RawVec", "MagDipole_Bfield", "MagDipole", "CircularLoop"]


class FDEM_CrossCheck(unittest.TestCase):
    if testBH:

        def test_BH_CrossCheck_jxr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["CurrentDensity", "x", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_jyr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["CurrentDensity", "y", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_jzr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["CurrentDensity", "z", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_jxi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["CurrentDensity", "x", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_jyi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["CurrentDensity", "y", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_jzi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["CurrentDensity", "z", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_exr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["ElectricField", "x", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_eyr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["ElectricField", "y", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_ezr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["ElectricField", "z", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_exi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["ElectricField", "x", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_eyi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["ElectricField", "y", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_ezi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["ElectricField", "z", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_bxr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticFluxDensity", "x", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_byr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticFluxDensity", "y", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_bzr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticFluxDensity", "z", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_bxi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticFluxDensity", "x", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_byi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticFluxDensity", "y", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_bzi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticFluxDensity", "z", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_hxr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticField", "x", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_hyr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticField", "y", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_hzr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticField", "z", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_hxi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticField", "x", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_hyi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticField", "y", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_hzi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticField", "z", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

    if testBH:

        def test_BH_CrossCheck_jxr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["CurrentDensity", "x", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_jyr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["CurrentDensity", "y", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_jzr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["CurrentDensity", "z", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_jxi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["CurrentDensity", "x", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_jyi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["CurrentDensity", "y", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_jzi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["CurrentDensity", "z", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_exr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["ElectricField", "x", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_eyr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["ElectricField", "y", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_ezr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["ElectricField", "z", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_exi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["ElectricField", "x", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_eyi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["ElectricField", "y", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_ezi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["ElectricField", "z", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_bxr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticFluxDensity", "x", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_byr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticFluxDensity", "y", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_bzr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticFluxDensity", "z", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_bxi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticFluxDensity", "x", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_byi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticFluxDensity", "y", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_bzi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticFluxDensity", "z", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_hxr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticField", "x", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_hyr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticField", "y", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_hzr(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticField", "z", "r"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_hxi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticField", "x", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_hyi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticField", "y", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )

        def test_BH_CrossCheck_hzi(self):
            self.assertTrue(
                crossCheckTest(
                    SrcList,
                    "b",
                    "h",
                    ["MagneticField", "z", "i"],
                    verbose=verbose,
                    TOL=TOLEJHB,
                )
            )