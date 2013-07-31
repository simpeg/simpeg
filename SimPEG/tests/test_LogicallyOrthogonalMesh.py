import numpy as np
import unittest
import sys
sys.path.append('../')
from TensorMesh import TensorMesh
from LogicallyOrthogonalMesh import LogicallyOrthogonalMesh
from OrderTest import OrderTest
from scipy.sparse.linalg import dsolve
from utils import ndgrid


class BasicLOMTests(unittest.TestCase):

    def setUp(self):
        a = np.array([1, 1, 1])
        b = np.array([1, 2])
        c = np.array([1, 4])
        gridIt = lambda h: [np.cumsum(np.r_[0, x]) for x in h]
        X, Y = ndgrid(gridIt([a, b]), vector=False)
        self.TM2 = TensorMesh([a, b])
        self.LOM2 = LogicallyOrthogonalMesh([X, Y])
        X, Y, Z = ndgrid(gridIt([a, b, c]), vector=False)
        self.TM3 = TensorMesh([a, b, c])
        self.LOM3 = LogicallyOrthogonalMesh([X, Y, Z])

    def test_area_3D(self):
        test_area = np.array([1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2])
        self.assertTrue(np.all(self.LOM3.area == test_area))

    def test_vol_3D(self):
        test_vol = np.array([1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8])
        np.testing.assert_almost_equal(self.LOM3.vol, test_vol)
        self.assertTrue(True)  # Pass if you get past the assertion.

    def test_vol_2D(self):
        test_vol = np.array([1, 1, 1, 2, 2, 2])
        t1 = np.all(self.LOM2.vol == test_vol)
        self.assertTrue(t1)

    def test_edge_3D(self):
        test_edge = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        t1 = np.all(self.LOM3.edge == test_edge)
        self.assertTrue(t1)

    def test_edge_2D(self):
        test_edge = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
        t1 = np.all(self.LOM2.edge == test_edge)
        self.assertTrue(t1)


if __name__ == '__main__':
    unittest.main()
