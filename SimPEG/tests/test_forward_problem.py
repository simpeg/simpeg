import numpy as np
import unittest
from SimPEG.mesh import TensorMesh
from SimPEG.forward import Problem
from TestUtils import checkDerivative
from scipy.sparse.linalg import dsolve


class ProblemTests(unittest.TestCase):

    def setUp(self):

        a = np.array([1, 1, 1])
        b = np.array([1, 2])
        c = np.array([1, 4])
        self.mesh2 = TensorMesh([a, b], np.array([3, 5]))
        self.p2 = Problem(self.mesh2)


    def test_modelTransform(self):
        print 'SimPEG.forward.Problem: Testing Model Transform'
        m = np.random.rand(self.mesh2.nC)
        passed = checkDerivative(lambda m : [self.p2.modelTransform(m), self.p2.modelTransformDeriv(m)], m, plotIt=False)
        self.assertTrue(passed)


if __name__ == '__main__':
    unittest.main()
