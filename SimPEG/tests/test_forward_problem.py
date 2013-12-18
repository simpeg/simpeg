import numpy as np
import unittest
from SimPEG import mesh, forward, inverse
from TestUtils import checkDerivative
from scipy.sparse.linalg import dsolve


class ProblemTests(unittest.TestCase):

    def setUp(self):

        a = np.array([1, 1, 1])
        b = np.array([1, 2])
        c = np.array([1, 4])
        self.mesh2 = mesh.TensorMesh([a, b], np.array([3, 5]))
        self.p2 = forward.Problem(self.mesh2)
        self.reg = inverse.Regularization(self.mesh2)

    def test_modelTransform(self):
        print 'SimPEG.forward.Problem: Testing Model Transform'
        m = np.random.rand(self.mesh2.nC)
        passed = checkDerivative(lambda m : [self.p2.modelTransform(m), self.p2.modelTransformDeriv(m)], m, plotIt=False)
        self.assertTrue(passed)

    def test_regularization(self):
        derChk = lambda m: [self.reg.modelObj(m), self.reg.modelObjDeriv(m)]
        mSynth = np.random.randn(self.mesh2.nC)
        checkDerivative(derChk, mSynth, plotIt=False)




if __name__ == '__main__':
    unittest.main()
