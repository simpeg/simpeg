from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import unittest
from SimPEG.electromagnetics import natural_source as nsem
from scipy.constants import mu_0


TOLr = 5e-2
TOL = 1e-4
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = [1e-1, 2e-1]
addrandoms = True


def JvecAdjointTest(inputSetup, comp="All", freq=False):
    (M, freqs, sig, sigBG, rx_loc) = inputSetup
    survey, problem = nsem.utils.test_utils.setupSimpegNSEM_ePrimSec(
        inputSetup, comp=comp, singleFreq=freq
    )
    print("Using {0} solver for the problem".format(problem.Solver))
    print(
        "Adjoint test of eForm primary/secondary "
        "for {:s} comp at {:s}\n".format(comp, str(survey.freqs))
    )

    m = sig
    u = problem.fields(m)
    np.random.seed(1983)
    v = np.random.rand(survey.nD,)
    # print problem.PropMap.PropModel.nP
    w = np.random.rand(problem.mesh.nC,)

    vJw = v.ravel().dot(problem.Jvec(m, w, u))
    wJtv = w.ravel().dot(problem.Jtvec(m, v, u))
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
        self.assertTrue(JvecAdjointTest(nsem.utils.test_utils.random(1e-2), "Imp", 0.1))


if __name__ == "__main__":
    unittest.main()
