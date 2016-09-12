# Test functions
import numpy as np
import SimPEG as simpeg
import unittest
from SimPEG import NSEM
from scipy.constants import mu_0


TOLr = 5e-2
TOL = 1e-4
FLR = 1e-20 # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = [1e-1, 2e-1]
addrandoms = True



def JvecAdjointTest(inputSetup,comp='All',freq=False):
    (M, freqs, sig, sigBG, rx_loc) = inputSetup
    survey, problem = NSEM.Utils.testUtils.setupSimpegNSEM_ePrimSec(inputSetup,comp=comp,singleFreq=freq)
    print 'Using {0} solver for the problem'.format(problem.Solver)
    print 'Adjoint test of eForm primary/secondary for {:s} comp at {:s}\n'.format(comp,str(survey.freqs))

    m  = sig
    u = problem.fields(m)

    v = np.random.rand(survey.nD,)
    # print problem.PropMap.PropModel.nP
    w = np.random.rand(problem.mesh.nC,)

    vJw = v.ravel().dot(problem.Jvec(m, w, u))
    wJtv = w.ravel().dot(problem.Jtvec(m, v, u))
    tol = np.max([TOL*(10**int(np.log10(np.abs(vJw)))),FLR])
    print ' vJw   wJtv  vJw - wJtv     tol    abs(vJw - wJtv) < tol'
    print vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol
    return np.abs(vJw - wJtv) < tol


class NSEM_AdjointTests(unittest.TestCase):

    def setUp(self):
        pass

    # Test the adjoint of Jvec and Jtvec
    def test_JvecAdjoint_zxxr(self):self.assertTrue(JvecAdjointTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zxxr',.1))
    def test_JvecAdjoint_zxxi(self):self.assertTrue(JvecAdjointTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zxxi',.1))
    def test_JvecAdjoint_zxyr(self):self.assertTrue(JvecAdjointTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zxyr',.1))
    def test_JvecAdjoint_zxyi(self):self.assertTrue(JvecAdjointTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zxyi',.1))
    def test_JvecAdjoint_zyxr(self):self.assertTrue(JvecAdjointTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zyxr',.1))
    def test_JvecAdjoint_zyxi(self):self.assertTrue(JvecAdjointTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zyxi',.1))
    def test_JvecAdjoint_zyyr(self):self.assertTrue(JvecAdjointTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zyyr',.1))
    def test_JvecAdjoint_zyyi(self):self.assertTrue(JvecAdjointTest(NSEM.Utils.testUtils.halfSpace(1e-2),'zyyi',.1))
    # def test_JvecAdjoint_All(self):self.assertTrue(JvecAdjointTest(NSEM.Utils.testUtils.random(1e-2),'Imp',.1))

if __name__ == '__main__':
    unittest.main()
