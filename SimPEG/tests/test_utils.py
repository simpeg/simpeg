import numpy as np
import unittest
import sys
sys.path.append('../')
from utils import mkvc, ndgrid, indexCube


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.a = np.array([1, 2, 3])
        self.b = np.array([1, 2])
        self.c = np.array([1, 2, 3, 4])

    def test_mkvc1(self):
        x = mkvc(self.a)
        self.assertTrue(x.shape, (3,))

    def test_mkvc2(self):
        x = mkvc(self.a, 2)
        self.assertTrue(x.shape, (3, 1))

    def test_mkvc3(self):
        x = mkvc(self.a, 3)
        self.assertTrue(x.shape, (3, 1, 1))

    def test_ndgrid_2D(self):
        XY = ndgrid([self.a, self.b])

        X1_test = np.array([1, 2, 3, 1, 2, 3])
        X2_test = np.array([1, 1, 1, 2, 2, 2])

        self.assertTrue(np.all(XY[:, 0] == X1_test))
        self.assertTrue(np.all(XY[:, 1] == X2_test))

    def test_ndgrid_3D(self):
        XYZ = ndgrid([self.a, self.b, self.c])

        X1_test = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
        X2_test = np.array([1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2])
        X3_test = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4])

        self.assertTrue(np.all(XYZ[:, 0] == X1_test))
        self.assertTrue(np.all(XYZ[:, 1] == X2_test))
        self.assertTrue(np.all(XYZ[:, 2] == X3_test))

    def test_indexCube_2D(self):
        nN = np.array([3, 3])
        self.assertTrue(np.all(indexCube('A', nN) == np.array([0, 1, 3, 4])))
        self.assertTrue(np.all(indexCube('B', nN) == np.array([3, 4, 6, 7])))
        self.assertTrue(np.all(indexCube('C', nN) == np.array([4, 5, 7, 8])))
        self.assertTrue(np.all(indexCube('D', nN) == np.array([1, 2, 4, 5])))

    def test_indexCube_3D(self):
        nN = np.array([3, 3, 3])
        self.assertTrue(np.all(indexCube('A', nN) == np.array([0, 1, 3, 4, 9, 10, 12, 13])))
        self.assertTrue(np.all(indexCube('B', nN) == np.array([3, 4, 6, 7, 12, 13, 15, 16])))
        self.assertTrue(np.all(indexCube('C', nN) == np.array([4, 5, 7, 8, 13, 14, 16, 17])))
        self.assertTrue(np.all(indexCube('D', nN) == np.array([1, 2, 4, 5, 10, 11, 13, 14])))
        self.assertTrue(np.all(indexCube('E', nN) == np.array([9, 10, 12, 13, 18, 19, 21, 22])))
        self.assertTrue(np.all(indexCube('F', nN) == np.array([12, 13, 15, 16, 21, 22, 24, 25])))
        self.assertTrue(np.all(indexCube('G', nN) == np.array([13, 14, 16, 17, 22, 23, 25, 26])))
        self.assertTrue(np.all(indexCube('H', nN) == np.array([10, 11, 13, 14, 19, 20, 22, 23])))

if __name__ == '__main__':
    unittest.main()
