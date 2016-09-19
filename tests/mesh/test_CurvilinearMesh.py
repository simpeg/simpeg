from __future__ import print_function
import numpy as np
import unittest
from SimPEG.Mesh import TensorMesh, CurvilinearMesh
from SimPEG.Utils import ndgrid


class BasicCurvTests(unittest.TestCase):

    def setUp(self):
        a = np.array([1, 1, 1])
        b = np.array([1, 2])
        c = np.array([1, 4])

        def gridIt(h): return [np.cumsum(np.r_[0, x]) for x in h]

        X, Y = ndgrid(gridIt([a, b]), vector=False)
        self.TM2 = TensorMesh([a, b])
        self.Curv2 = CurvilinearMesh([X, Y])
        X, Y, Z = ndgrid(gridIt([a, b, c]), vector=False)
        self.TM3 = TensorMesh([a, b, c])
        self.Curv3 = CurvilinearMesh([X, Y, Z])

    def test_area_3D(self):
        test_area = np.array([1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1,
                              1, 2, 2, 2])
        self.assertTrue(np.all(self.Curv3.area == test_area))

    def test_vol_3D(self):
        test_vol = np.array([1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8])
        np.testing.assert_almost_equal(self.Curv3.vol, test_vol)
        self.assertTrue(True)  # Pass if you get past the assertion.

    def test_vol_2D(self):
        test_vol = np.array([1, 1, 1, 2, 2, 2])
        t1 = np.all(self.Curv2.vol == test_vol)
        self.assertTrue(t1)

    def test_edge_3D(self):
        test_edge = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                              2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2,
                              2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        t1 = np.all(self.Curv3.edge == test_edge)
        self.assertTrue(t1)

    def test_edge_2D(self):
        test_edge = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                              2])
        t1 = np.all(self.Curv2.edge == test_edge)
        self.assertTrue(t1)

    def test_tangents(self):
        T = self.Curv2.tangents
        self.assertTrue(np.all(self.Curv2.r(T, 'E', 'Ex', 'V')[0] ==
                        np.ones(self.Curv2.nEx)))
        self.assertTrue(np.all(self.Curv2.r(T, 'E', 'Ex', 'V')[1] ==
                        np.zeros(self.Curv2.nEx)))
        self.assertTrue(np.all(self.Curv2.r(T, 'E', 'Ey', 'V')[0] ==
                        np.zeros(self.Curv2.nEy)))
        self.assertTrue(np.all(self.Curv2.r(T, 'E', 'Ey', 'V')[1] ==
                        np.ones(self.Curv2.nEy)))

        T = self.Curv3.tangents
        self.assertTrue(np.all(self.Curv3.r(T, 'E', 'Ex', 'V')[0] ==
                        np.ones(self.Curv3.nEx)))
        self.assertTrue(np.all(self.Curv3.r(T, 'E', 'Ex', 'V')[1] ==
                        np.zeros(self.Curv3.nEx)))
        self.assertTrue(np.all(self.Curv3.r(T, 'E', 'Ex', 'V')[2] ==
                        np.zeros(self.Curv3.nEx)))

        self.assertTrue(np.all(self.Curv3.r(T, 'E', 'Ey', 'V')[0] ==
                        np.zeros(self.Curv3.nEy)))
        self.assertTrue(np.all(self.Curv3.r(T, 'E', 'Ey', 'V')[1] ==
                        np.ones(self.Curv3.nEy)))
        self.assertTrue(np.all(self.Curv3.r(T, 'E', 'Ey', 'V')[2] ==
                        np.zeros(self.Curv3.nEy)))

        self.assertTrue(np.all(self.Curv3.r(T, 'E', 'Ez', 'V')[0] ==
                        np.zeros(self.Curv3.nEz)))
        self.assertTrue(np.all(self.Curv3.r(T, 'E', 'Ez', 'V')[1] ==
                        np.zeros(self.Curv3.nEz)))
        self.assertTrue(np.all(self.Curv3.r(T, 'E', 'Ez', 'V')[2] ==
                        np.ones(self.Curv3.nEz)))

    def test_normals(self):
        N = self.Curv2.normals
        self.assertTrue(np.all(self.Curv2.r(N, 'F', 'Fx', 'V')[0] ==
                        np.ones(self.Curv2.nFx)))
        self.assertTrue(np.all(self.Curv2.r(N, 'F', 'Fx', 'V')[1] ==
                        np.zeros(self.Curv2.nFx)))
        self.assertTrue(np.all(self.Curv2.r(N, 'F', 'Fy', 'V')[0] ==
                        np.zeros(self.Curv2.nFy)))
        self.assertTrue(np.all(self.Curv2.r(N, 'F', 'Fy', 'V')[1] ==
                        np.ones(self.Curv2.nFy)))

        N = self.Curv3.normals
        self.assertTrue(np.all(self.Curv3.r(N, 'F', 'Fx', 'V')[0] ==
                        np.ones(self.Curv3.nFx)))
        self.assertTrue(np.all(self.Curv3.r(N, 'F', 'Fx', 'V')[1] ==
                        np.zeros(self.Curv3.nFx)))
        self.assertTrue(np.all(self.Curv3.r(N, 'F', 'Fx', 'V')[2] ==
                        np.zeros(self.Curv3.nFx)))

        self.assertTrue(np.all(self.Curv3.r(N, 'F', 'Fy', 'V')[0] ==
                        np.zeros(self.Curv3.nFy)))
        self.assertTrue(np.all(self.Curv3.r(N, 'F', 'Fy', 'V')[1] ==
                        np.ones(self.Curv3.nFy)))
        self.assertTrue(np.all(self.Curv3.r(N, 'F', 'Fy', 'V')[2] ==
                        np.zeros(self.Curv3.nFy)))

        self.assertTrue(np.all(self.Curv3.r(N, 'F', 'Fz', 'V')[0] ==
                        np.zeros(self.Curv3.nFz)))
        self.assertTrue(np.all(self.Curv3.r(N, 'F', 'Fz', 'V')[1] ==
                        np.zeros(self.Curv3.nFz)))
        self.assertTrue(np.all(self.Curv3.r(N, 'F', 'Fz', 'V')[2] ==
                        np.ones(self.Curv3.nFz)))

    def test_grid(self):
        self.assertTrue(np.all(self.Curv2.gridCC == self.TM2.gridCC))
        self.assertTrue(np.all(self.Curv2.gridN == self.TM2.gridN))
        self.assertTrue(np.all(self.Curv2.gridFx == self.TM2.gridFx))
        self.assertTrue(np.all(self.Curv2.gridFy == self.TM2.gridFy))
        self.assertTrue(np.all(self.Curv2.gridEx == self.TM2.gridEx))
        self.assertTrue(np.all(self.Curv2.gridEy == self.TM2.gridEy))

        self.assertTrue(np.all(self.Curv3.gridCC == self.TM3.gridCC))
        self.assertTrue(np.all(self.Curv3.gridN == self.TM3.gridN))
        self.assertTrue(np.all(self.Curv3.gridFx == self.TM3.gridFx))
        self.assertTrue(np.all(self.Curv3.gridFy == self.TM3.gridFy))
        self.assertTrue(np.all(self.Curv3.gridFz == self.TM3.gridFz))
        self.assertTrue(np.all(self.Curv3.gridEx == self.TM3.gridEx))
        self.assertTrue(np.all(self.Curv3.gridEy == self.TM3.gridEy))
        self.assertTrue(np.all(self.Curv3.gridEz == self.TM3.gridEz))


if __name__ == '__main__':
    unittest.main()
