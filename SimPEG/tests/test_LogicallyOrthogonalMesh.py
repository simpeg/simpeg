import numpy as np
import unittest
import sys
sys.path.append('../')
from TensorMesh import TensorMesh
from LogicallyOrthogonalMesh import LogicallyOrthogonalMesh
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

    def test_tangents(self):
        T = self.LOM2.tangents
        self.assertTrue(np.all(self.LOM2.r(T, 'E', 'Ex', 'V')[0] == np.ones(self.LOM2.nE[0])))
        self.assertTrue(np.all(self.LOM2.r(T, 'E', 'Ex', 'V')[1] == np.zeros(self.LOM2.nE[0])))
        self.assertTrue(np.all(self.LOM2.r(T, 'E', 'Ey', 'V')[0] == np.zeros(self.LOM2.nE[1])))
        self.assertTrue(np.all(self.LOM2.r(T, 'E', 'Ey', 'V')[1] == np.ones(self.LOM2.nE[1])))

        T = self.LOM3.tangents
        self.assertTrue(np.all(self.LOM3.r(T, 'E', 'Ex', 'V')[0] == np.ones(self.LOM3.nE[0])))
        self.assertTrue(np.all(self.LOM3.r(T, 'E', 'Ex', 'V')[1] == np.zeros(self.LOM3.nE[0])))
        self.assertTrue(np.all(self.LOM3.r(T, 'E', 'Ex', 'V')[2] == np.zeros(self.LOM3.nE[0])))

        self.assertTrue(np.all(self.LOM3.r(T, 'E', 'Ey', 'V')[0] == np.zeros(self.LOM3.nE[1])))
        self.assertTrue(np.all(self.LOM3.r(T, 'E', 'Ey', 'V')[1] == np.ones(self.LOM3.nE[1])))
        self.assertTrue(np.all(self.LOM3.r(T, 'E', 'Ey', 'V')[2] == np.zeros(self.LOM3.nE[1])))

        self.assertTrue(np.all(self.LOM3.r(T, 'E', 'Ez', 'V')[0] == np.zeros(self.LOM3.nE[2])))
        self.assertTrue(np.all(self.LOM3.r(T, 'E', 'Ez', 'V')[1] == np.zeros(self.LOM3.nE[2])))
        self.assertTrue(np.all(self.LOM3.r(T, 'E', 'Ez', 'V')[2] == np.ones(self.LOM3.nE[2])))

    def test_normals(self):
        N = self.LOM2.normals
        self.assertTrue(np.all(self.LOM2.r(N, 'F', 'Fx', 'V')[0] == np.ones(self.LOM2.nF[0])))
        self.assertTrue(np.all(self.LOM2.r(N, 'F', 'Fx', 'V')[1] == np.zeros(self.LOM2.nF[0])))
        self.assertTrue(np.all(self.LOM2.r(N, 'F', 'Fy', 'V')[0] == np.zeros(self.LOM2.nF[1])))
        self.assertTrue(np.all(self.LOM2.r(N, 'F', 'Fy', 'V')[1] == np.ones(self.LOM2.nF[1])))

        N = self.LOM3.normals
        self.assertTrue(np.all(self.LOM3.r(N, 'F', 'Fx', 'V')[0] == np.ones(self.LOM3.nF[0])))
        self.assertTrue(np.all(self.LOM3.r(N, 'F', 'Fx', 'V')[1] == np.zeros(self.LOM3.nF[0])))
        self.assertTrue(np.all(self.LOM3.r(N, 'F', 'Fx', 'V')[2] == np.zeros(self.LOM3.nF[0])))

        self.assertTrue(np.all(self.LOM3.r(N, 'F', 'Fy', 'V')[0] == np.zeros(self.LOM3.nF[1])))
        self.assertTrue(np.all(self.LOM3.r(N, 'F', 'Fy', 'V')[1] == np.ones(self.LOM3.nF[1])))
        self.assertTrue(np.all(self.LOM3.r(N, 'F', 'Fy', 'V')[2] == np.zeros(self.LOM3.nF[1])))

        self.assertTrue(np.all(self.LOM3.r(N, 'F', 'Fz', 'V')[0] == np.zeros(self.LOM3.nF[2])))
        self.assertTrue(np.all(self.LOM3.r(N, 'F', 'Fz', 'V')[1] == np.zeros(self.LOM3.nF[2])))
        self.assertTrue(np.all(self.LOM3.r(N, 'F', 'Fz', 'V')[2] == np.ones(self.LOM3.nF[2])))

    def test_grid(self):
        self.assertTrue(np.all(self.LOM2.gridFx == self.TM2.gridFx))
        self.assertTrue(np.all(self.LOM2.gridFy == self.TM2.gridFy))
        self.assertTrue(np.all(self.LOM2.gridEx == self.TM2.gridEx))
        self.assertTrue(np.all(self.LOM2.gridEy == self.TM2.gridEy))

        self.assertTrue(np.all(self.LOM3.gridFx == self.TM3.gridFx))
        self.assertTrue(np.all(self.LOM3.gridFy == self.TM3.gridFy))
        self.assertTrue(np.all(self.LOM3.gridFz == self.TM3.gridFz))
        self.assertTrue(np.all(self.LOM3.gridEx == self.TM3.gridEx))
        self.assertTrue(np.all(self.LOM3.gridEy == self.TM3.gridEy))
        self.assertTrue(np.all(self.LOM3.gridEz == self.TM3.gridEz))


if __name__ == '__main__':
    unittest.main()
