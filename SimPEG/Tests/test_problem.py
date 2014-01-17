import numpy as np
import unittest
from SimPEG import *
from TestUtils import checkDerivative
from scipy.sparse.linalg import dsolve


class ProblemTests(unittest.TestCase):

    def setUp(self):

        a = np.array([1, 1, 1])
        b = np.array([1, 2])
        c = np.array([1, 4])
        self.mesh2 = Mesh.TensorMesh([a, b], np.array([3, 5]))
        self.p2 = Problem.BaseProblem(self.mesh2, None)
        self.reg = Regularization.BaseRegularization(self.mesh2)

    def test_regularization(self):
        derChk = lambda m: [self.reg.modelObj(m), self.reg.modelObjDeriv(m)]
        mSynth = np.random.randn(self.mesh2.nC)
        checkDerivative(derChk, mSynth, plotIt=False)


if __name__ == '__main__':
    unittest.main()
