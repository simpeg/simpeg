import numpy as np
import unittest
import sys
sys.path.append('../')
from TensorMesh import TensorMesh


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        a = np.array([1, 1, 1])
        b = np.array([1, 2])
        c = np.array([1, 4])
        self.mesh2 = TensorMesh([a, b], np.array([3, 5]))
        self.mesh3 = TensorMesh([a, b, c])

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

    def test_area_3D(self):
        test_area = np.array([1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2])
        t1 = np.all(self.mesh3.area == test_area)
        self.assertTrue(t1)

    def test_vol_3D(self):
        test_vol = np.array([1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8])
        t1 = np.all(self.mesh3.vol == test_vol)
        self.assertTrue(t1)

    def test_vol_2D(self):
        test_vol = np.array([1, 1, 1, 2, 2, 2])
        t1 = np.all(self.mesh2.vol == test_vol)
        self.assertTrue(t1)

    def test_edge_3D(self):
        test_edge = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        t1 = np.all(self.mesh3.edge == test_edge)
        self.assertTrue(t1)

    def test_edge_2D(self):
        test_edge = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
        t1 = np.all(self.mesh2.edge == test_edge)
        self.assertTrue(t1)


if __name__ == '__main__':
    unittest.main()
