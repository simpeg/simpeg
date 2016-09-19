# Test functions
from glob import glob
import numpy as np, sys, os, time, scipy, subprocess
import SimPEG as simpeg
import unittest
from SimPEG import NSEM
from SimPEG.Utils import meshTensor
from scipy.constants import mu_0


TOLr = 5e-2
TOL = 1e-4
FLR = 1e-20 # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = [1e-1, 2e-1]
addrandoms = True



def JvecAdjointTest(sigmaHalf,formulation='PrimSec'):
    forType = 'PrimSec' not in formulation
    survey, sigma, sigBG, m1d = NSEM.Utils.testUtils.setup1DSurvey(sigmaHalf,tD=forType,structure=False)
    print 'Adjoint test of e formulation for {:s} comp \n'.format(formulation)

    if 'PrimSec' in formulation:
        problem = NSEM.Problem1D_ePrimSec(m1d, sigmaPrimary = sigBG)
    else:
        problem = NSEM.Problem1D_eTotal(m1d)
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
    print ' vJw   wJtv  vJw - wJtv     tol    abs(vJw - wJtv) < tol'
    print vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol
    return np.abs(vJw - wJtv) < tol


class NSEM_1D_AdjointTests(unittest.TestCase):

    def setUp(self):
        pass

    # Test the adjoint of Jvec and Jtvec
<<<<<<< HEAD:tests/em/nsem/inversion/test_Problem1D_Adjoint.py
    # def test_JvecAdjoint_zxxr(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zxxr',.1))
    # def test_JvecAdjoint_zxxi(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zxxi',.1))
    # def test_JvecAdjoint_zxyr(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zxyr',.1))
    # def test_JvecAdjoint_zxyi(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zxyi',.1))
    # def test_JvecAdjoint_zyxr(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zyxr',.1))
    # def test_JvecAdjoint_zyxi(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zyxi',.1))
    # def test_JvecAdjoint_zyyr(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zyyr',.1))
    # def test_JvecAdjoint_zyyi(self):self.assertTrue(JvecAdjointTest(random(1e-2),'zyyi',.1))
    def test_JvecAdjoint_All(self):self.assertTrue(JvecAdjointTest(1e-2))
=======
    def test_JvecAdjoint_zxxr(self):self.assertTrue(JvecAdjointTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zxxr',.1))
    def test_JvecAdjoint_zxxi(self):self.assertTrue(JvecAdjointTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zxxi',.1))
    def test_JvecAdjoint_zxyr(self):self.assertTrue(JvecAdjointTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zxyr',.1))
    def test_JvecAdjoint_zxyi(self):self.assertTrue(JvecAdjointTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zxyi',.1))
    def test_JvecAdjoint_zyxr(self):self.assertTrue(JvecAdjointTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zyxr',.1))
    def test_JvecAdjoint_zyxi(self):self.assertTrue(JvecAdjointTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zyxi',.1))
    def test_JvecAdjoint_zyyr(self):self.assertTrue(JvecAdjointTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zyyr',.1))
    def test_JvecAdjoint_zyyi(self):self.assertTrue(JvecAdjointTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zyyi',.1))
    # def test_JvecAdjoint_All(self):self.assertTrue(JvecAdjointTest(NSEM.Utils.testUtils.random(1e-2),'All',.1))
>>>>>>> mt/dev:tests/mt/inversion/test_Problem3D_Adjoint.py

if __name__ == '__main__':
    unittest.main()
