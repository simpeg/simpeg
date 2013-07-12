import numpy as np
import unittest
import sys
sys.path.append('../')
from TensorMesh import TensorMesh


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        a = np.array([1, 1, 1])
        b = np.array([1, 2])
        x0 = np.array([3, 5])
        self.mesh2 = TensorMesh([a, b], x0)

    def test_vectorN_2D(self):
        testNx = np.array([3, 4, 5, 6])
        testNy = np.array([5, 6, 8])

        xtest = np.all(self.mesh2.vectorNx == testNx)
        ytest = np.all(self.mesh2.vectorNy == testNy)
        self.assertTrue(xtest and ytest)

    def test_vectorCC_2D(self):
        testNx = np.array([3.5, 4.5, 5.5])
        testNy = np.array([5.5, 7])

        xtest = np.all(self.mesh2.vectorCCx == testNx)
        ytest = np.all(self.mesh2.vectorCCy == testNy)
        self.assertTrue(xtest and ytest)


if __name__ == '__main__':
    unittest.main()
