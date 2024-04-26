import numpy as np
import unittest
from simpeg.electromagnetics import natural_source as nsem
from scipy.constants import mu_0


TOLr = 5e-2
TOL = 1e-4
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = [1e-1, 2e-1]
addrandoms = True


def JvecAdjointTest(
    inputSetup, comp="All", freq=False, testLocations=False, testSingle=False
):
    if testLocations:
        if testSingle:
            (
                m,
                simulation,
            ) = nsem.utils.test_utils.setupSimpegNSEM_tests_location_assign_list(
                inputSetup, [freq], comp=comp, singleFreq=False, singleList=True
            )
        else:
            (
                m,
                simulation,
            ) = nsem.utils.test_utils.setupSimpegNSEM_tests_location_assign_list(
                inputSetup, [freq], comp=comp, singleFreq=False
            )
            # print(simulation.)
    else:
        m, simulation = nsem.utils.test_utils.setupSimpegNSEM_PrimarySecondary(
            inputSetup, [freq], comp=comp, singleFreq=False
        )

    print("Using {0} solver for the simulation".format(simulation.solver))
    print(
        "Adjoint test of eForm primary/secondary "
        "for {:s} comp at {:s}\n".format(comp, str(simulation.survey.frequencies))
    )

    u = simulation.fields(m)
    np.random.seed(1983)
    v = np.random.rand(
        simulation.survey.nD,
    )
    # print problem.PropMap.PropModel.nP
    w = np.random.rand(
        len(m),
    )
    # print(problem.Jvec(m, w, u))
    vJw = v.ravel().dot(simulation.Jvec(m, w, u))
    wJtv = w.ravel().dot(simulation.Jtvec(m, v, u))
    tol = np.max([TOL * (10 ** int(np.log10(np.abs(vJw)))), FLR])
    print(" vJw   wJtv  vJw - wJtv     tol    abs(vJw - wJtv) < tol")
    print(vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol)
    return np.abs(vJw - wJtv) < tol


class NSEM_3D_AdjointTests(unittest.TestCase):
    # Test the adjoint of Jvec and Jtvec
    def test_JvecAdjoint_zxx(self):
        self.assertTrue(
            JvecAdjointTest(nsem.utils.test_utils.halfSpace(1e-2), "xx", 0.1)
        )

    def test_JvecAdjoint_zxy(self):
        self.assertTrue(
            JvecAdjointTest(nsem.utils.test_utils.halfSpace(1e-2), "xy", 0.1)
        )

    def test_JvecAdjoint_zyx(self):
        self.assertTrue(
            JvecAdjointTest(nsem.utils.test_utils.halfSpace(1e-2), "yx", 0.1)
        )

    def test_JvecAdjoint_zyy(self):
        self.assertTrue(
            JvecAdjointTest(nsem.utils.test_utils.halfSpace(1e-2), "yy", 0.1)
        )

    def test_JvecAdjoint_tzx(self):
        self.assertTrue(
            JvecAdjointTest(nsem.utils.test_utils.halfSpace(1e-2), "zx", 0.1)
        )

    def test_JvecAdjoint_tzy(self):
        self.assertTrue(
            JvecAdjointTest(nsem.utils.test_utils.halfSpace(1e-2), "zy", 0.1)
        )

    def test_JvecAdjoint_All(self):
        self.assertTrue(JvecAdjointTest(nsem.utils.test_utils.random(1e-2), "All", 0.1))

    def test_JvecAdjoint_Imp(self):
        self.assertTrue(JvecAdjointTest(nsem.utils.test_utils.random(1e-2), "Imp", 0.1))

    def test_JvecAdjoint_Res(self):
        self.assertTrue(JvecAdjointTest(nsem.utils.test_utils.random(1e-2), "Res", 0.1))

    # test location assign
    def test_JvecAdjoint_location_e_b(self):
        self.assertTrue(
            JvecAdjointTest(
                nsem.utils.test_utils.random(1e-2),
                "Res",
                0.1,
                testLocations=True,
                testSingle=False,
            )
        )

    def test_JvecAdjoint_location_single(self):
        self.assertTrue(
            JvecAdjointTest(
                nsem.utils.test_utils.random(1e-2),
                "Res",
                0.1,
                testLocations=True,
                testSingle=True,
            )
        )

    def test_JvecAdjoint_location_single_all(self):
        self.assertTrue(
            JvecAdjointTest(
                nsem.utils.test_utils.random(1e-2),
                "All",
                0.1,
                testLocations=True,
                testSingle=True,
            )
        )

    def test_JvecAdjoint_location_single_imp(self):
        self.assertTrue(
            JvecAdjointTest(
                nsem.utils.test_utils.random(1e-2),
                "Imp",
                0.1,
                testLocations=True,
                testSingle=True,
            )
        )

    def test_JvecAdjoint_location_single_tip(self):
        self.assertTrue(
            JvecAdjointTest(
                nsem.utils.test_utils.random(1e-2),
                "Tip",
                0.1,
                testLocations=True,
                testSingle=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
