from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import unittest
from SimPEG.EM import NSEM
from scipy.constants import mu_0


TOLr = 5e-2
TOL = 1e-4
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = [1e-1, 2e-1]
addrandoms = True


def JvecAdjointTest(inputSetup, tf_type='All', freq=False):
    (M, freqs, sig, sigBG, rx_loc) = inputSetup
    survey, problem = NSEM.Utils.testUtils.setupSimpegNSEM_ePrimSec(
        inputSetup, tf_type=tf_type, singleFreq=freq
    )
    print('Using {0} solver for the problem'.format(problem.Solver))
    print(
        'Adjoint test of eForm primary/secondary '
        'for {:s} transfer function at {:s} Hz\n'.format(tf_type, str(survey.freqs))
    )

    m = sig
    u = problem.fields(m)
    np.random.seed(1983)
    v = np.random.rand(survey.nD,)
    # print problem.PropMap.PropModel.nP
    w = np.random.rand(problem.mesh.nC,)

    vJw = v.ravel().dot(problem.Jvec(m, w, u))
    wJtv = w.ravel().dot(problem.Jtvec(m, v, u))
    tol = np.max([TOL*(10**int(np.log10(np.abs(vJw)))), FLR])
    print(' vJw   wJtv  vJw - wJtv     tol    abs(vJw - wJtv) < tol')
    print(vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol)
    return np.abs(vJw - wJtv) < tol


class NSEM_3D_AdjointTests(unittest.TestCase):

    # Test the adjoint of Jvec and Jtvec
    def test_JvecAdjoint_Imp(self):
        self.assertTrue(JvecAdjointTest(
            NSEM.Utils.testUtils.blockInhalfSpace([1e-2, 1]), 'Imp', .1))
    def test_JvecAdjoint_Tip(self):
        self.assertTrue(JvecAdjointTest(
            NSEM.Utils.testUtils.blockInhalfSpace([1e-2, 1]), 'Tip', .1))
    def test_JvecAdjoint_HMV(self):
        self.assertTrue(JvecAdjointTest(
            NSEM.Utils.testUtils.blockInhalfSpace([1e-2, 1]), 'HMV', .1))
if __name__ == '__main__':
    unittest.main()
