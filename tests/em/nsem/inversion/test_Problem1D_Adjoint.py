from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import unittest
from scipy.constants import mu_0

from SimPEG.EM import NSEM
from SimPEG import Maps


TOL = 1e-4
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0


def JvecAdjointTest(sigmaHalf, formulation='PrimSec'):
    forType = 'PrimSec' not in formulation
    survey, sigma, sigBG, m1d = NSEM.Utils.testUtils.setup1DSurvey(sigmaHalf,tD=forType,structure=False)
    print('Adjoint test of e formulation for {:s} comp \n'.format(formulation))

    if 'PrimSec' in formulation:
        problem = NSEM.Problem1D_ePrimSec(m1d, sigmaPrimary=sigBG, sigmaMap=Maps.IdentityMap(m1d))
    else:
        raise NotImplementedError('Only {} formulations are implemented.'.format(formulation))
    problem.pair(survey)
    m  = sigma
    u = problem.fields(m)

    np.random.seed(1983)
    v = np.random.rand(survey.nD,)
    # print problem.PropMap.PropModel.nP
    w = np.random.rand(problem.mesh.nC,)

    vJw = v.ravel().dot(problem.Jvec(m, w, u))
    wJtv = w.ravel().dot(problem.Jtvec(m, v, u))
    tol = np.max([TOL*(10**int(np.log10(np.abs(vJw)))),FLR])
    print(' vJw   wJtv  vJw - wJtv     tol    abs(vJw - wJtv) < tol')
    print(vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol)
    return np.abs(vJw - wJtv) < tol


class NSEM_1D_AdjointTests(unittest.TestCase):

    def setUp(self):
        pass

    # Test the adjoint of Jvec and Jtvec
    # def test_JvecAdjoint_zxxr(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zxxr',.1))
    # def test_JvecAdjoint_zxxi(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zxxi',.1))
    # def test_JvecAdjoint_zxyr(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zxyr',.1))
    # def test_JvecAdjoint_zxyi(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zxyi',.1))
    # def test_JvecAdjoint_zyxr(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zyxr',.1))
    # def test_JvecAdjoint_zyxi(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zyxi',.1))
    # def test_JvecAdjoint_zyyr(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zyyr',.1))
    # def test_JvecAdjoint_zyyi(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zyyi',.1))
    def test_JvecAdjoint_All(self):self.assertTrue(JvecAdjointTest(1e-2))


if __name__ == '__main__':
    unittest.main()
