from __future__ import print_function
import unittest
from SimPEG.Mesh.BaseMesh import BaseRectangularMesh
import numpy as np


class TestBaseMesh(unittest.TestCase):

    def setUp(self):
        self.mesh = BaseRectangularMesh([6, 2, 3])

    def test_meshDimensions(self):
        self.assertTrue(self.mesh.dim, 3)

    def test_mesh_nc(self):
        self.assertTrue(self.mesh.nC == 36)
        self.assertTrue(np.all(self.mesh.vnC == [6, 2, 3]))

    def test_mesh_nc_xyz(self):
        self.assertTrue(np.all(self.mesh.nCx == 6))
        self.assertTrue(np.all(self.mesh.nCy == 2))
        self.assertTrue(np.all(self.mesh.nCz == 3))

    def test_mesh_nf(self):
        self.assertTrue(np.all(self.mesh.vnFx == [7, 2, 3]))
        self.assertTrue(np.all(self.mesh.vnFy == [6, 3, 3]))
        self.assertTrue(np.all(self.mesh.vnFz == [6, 2, 4]))

    def test_mesh_ne(self):
        self.assertTrue(np.all(self.mesh.vnEx == [6, 3, 4]))
        self.assertTrue(np.all(self.mesh.vnEy == [7, 2, 4]))
        self.assertTrue(np.all(self.mesh.vnEz == [7, 3, 3]))

    def test_mesh_numbers(self):
        self.assertTrue(self.mesh.nC == 36)
        self.assertTrue(np.all(self.mesh.vnF == [42, 54, 48]))
        self.assertTrue(np.all(self.mesh.vnE == [72, 56, 63]))
        self.assertTrue(np.all(self.mesh.nF == np.sum([42, 54, 48])))
        self.assertTrue(np.all(self.mesh.nE == np.sum([72, 56, 63])))

    def test_mesh_r_E_V(self):
        ex = np.ones(self.mesh.nEx)
        ey = np.ones(self.mesh.nEy)*2
        ez = np.ones(self.mesh.nEz)*3
        e = np.r_[ex, ey, ez]
        tex = self.mesh.r(e, 'E', 'Ex', 'V')
        tey = self.mesh.r(e, 'E', 'Ey', 'V')
        tez = self.mesh.r(e, 'E', 'Ez', 'V')
        self.assertTrue(np.all(tex == ex))
        self.assertTrue(np.all(tey == ey))
        self.assertTrue(np.all(tez == ez))
        tex, tey, tez = self.mesh.r(e, 'E', 'E', 'V')
        self.assertTrue(np.all(tex == ex))
        self.assertTrue(np.all(tey == ey))
        self.assertTrue(np.all(tez == ez))

    def test_mesh_r_F_V(self):
        fx = np.ones(self.mesh.nFx)
        fy = np.ones(self.mesh.nFy)*2
        fz = np.ones(self.mesh.nFz)*3
        f = np.r_[fx, fy, fz]
        tfx = self.mesh.r(f, 'F', 'Fx', 'V')
        tfy = self.mesh.r(f, 'F', 'Fy', 'V')
        tfz = self.mesh.r(f, 'F', 'Fz', 'V')
        self.assertTrue(np.all(tfx == fx))
        self.assertTrue(np.all(tfy == fy))
        self.assertTrue(np.all(tfz == fz))
        tfx, tfy, tfz = self.mesh.r(f, 'F', 'F', 'V')
        self.assertTrue(np.all(tfx == fx))
        self.assertTrue(np.all(tfy == fy))
        self.assertTrue(np.all(tfz == fz))

    def test_mesh_r_E_M(self):
        g = np.ones((np.prod(self.mesh.vnEx), 3))
        g[:, 1] = 2
        g[:, 2] = 3
        Xex, Yex, Zex = self.mesh.r(g, 'Ex', 'Ex', 'M')
        self.assertTrue(np.all(Xex.shape == self.mesh.vnEx))
        self.assertTrue(np.all(Yex.shape == self.mesh.vnEx))
        self.assertTrue(np.all(Zex.shape == self.mesh.vnEx))
        self.assertTrue(np.all(Xex == 1))
        self.assertTrue(np.all(Yex == 2))
        self.assertTrue(np.all(Zex == 3))

    def test_mesh_r_F_M(self):
        g = np.ones((np.prod(self.mesh.vnFx), 3))
        g[:, 1] = 2
        g[:, 2] = 3
        Xfx, Yfx, Zfx = self.mesh.r(g, 'Fx', 'Fx', 'M')
        self.assertTrue(np.all(Xfx.shape == self.mesh.vnFx))
        self.assertTrue(np.all(Yfx.shape == self.mesh.vnFx))
        self.assertTrue(np.all(Zfx.shape == self.mesh.vnFx))
        self.assertTrue(np.all(Xfx == 1))
        self.assertTrue(np.all(Yfx == 2))
        self.assertTrue(np.all(Zfx == 3))

    def test_mesh_r_CC_M(self):
        g = np.ones((self.mesh.nC, 3))
        g[:, 1] = 2
        g[:, 2] = 3
        Xc, Yc, Zc = self.mesh.r(g, 'CC', 'CC', 'M')
        self.assertTrue(np.all(Xc.shape == self.mesh.vnC))
        self.assertTrue(np.all(Yc.shape == self.mesh.vnC))
        self.assertTrue(np.all(Zc.shape == self.mesh.vnC))
        self.assertTrue(np.all(Xc == 1))
        self.assertTrue(np.all(Yc == 2))
        self.assertTrue(np.all(Zc == 3))


class TestMeshNumbers2D(unittest.TestCase):

    def setUp(self):
        self.mesh = BaseRectangularMesh([6, 2])

    def test_meshDimensions(self):
        self.assertTrue(self.mesh.dim, 2)

    def test_mesh_nc(self):
        self.assertTrue(np.all(self.mesh.vnC == [6, 2]))

    def test_mesh_nc_xyz(self):
        self.assertTrue(np.all(self.mesh.nCx == 6))
        self.assertTrue(np.all(self.mesh.nCy == 2))
        self.assertTrue(self.mesh.nCz is None)

    def test_mesh_nf(self):
        self.assertTrue(np.all(self.mesh.vnFx == [7, 2]))
        self.assertTrue(np.all(self.mesh.vnFy == [6, 3]))
        self.assertTrue(self.mesh.vnFz is None)

    def test_mesh_ne(self):
        self.assertTrue(np.all(self.mesh.vnEx == [6, 3]))
        self.assertTrue(np.all(self.mesh.vnEy == [7, 2]))
        self.assertTrue(self.mesh.vnEz is None)

    def test_mesh_numbers(self):
        c = self.mesh.nC == 12
        self.assertTrue(np.all(self.mesh.vnF == [14, 18]))
        self.assertTrue(np.all(self.mesh.nFx == 14))
        self.assertTrue(np.all(self.mesh.nFy == 18))
        self.assertTrue(np.all(self.mesh.nEx == 18))
        self.assertTrue(np.all(self.mesh.nEy == 14))
        self.assertTrue(np.all(self.mesh.vnE == [18, 14]))
        self.assertTrue(np.all(self.mesh.vnE == [18, 14]))
        self.assertTrue(np.all(self.mesh.nF == np.sum([14, 18])))
        self.assertTrue(np.all(self.mesh.nE == np.sum([18, 14])))

    def test_mesh_r_E_V(self):
        ex = np.ones(self.mesh.nEx)
        ey = np.ones(self.mesh.nEy)*2
        e = np.r_[ex, ey]
        tex = self.mesh.r(e, 'E', 'Ex', 'V')
        tey = self.mesh.r(e, 'E', 'Ey', 'V')
        self.assertTrue(np.all(tex == ex))
        self.assertTrue(np.all(tey == ey))
        tex, tey = self.mesh.r(e, 'E', 'E', 'V')
        self.assertTrue(np.all(tex == ex))
        self.assertTrue(np.all(tey == ey))
        self.assertRaises(AssertionError,   self.mesh.r, e, 'E', 'Ez', 'V')

    def test_mesh_r_F_V(self):
        fx = np.ones(self.mesh.nFx)
        fy = np.ones(self.mesh.nFy)*2
        f = np.r_[fx, fy]
        tfx = self.mesh.r(f, 'F', 'Fx', 'V')
        tfy = self.mesh.r(f, 'F', 'Fy', 'V')
        self.assertTrue(np.all(tfx == fx))
        self.assertTrue(np.all(tfy == fy))
        tfx, tfy = self.mesh.r(f, 'F', 'F', 'V')
        self.assertTrue(np.all(tfx == fx))
        self.assertTrue(np.all(tfy == fy))
        self.assertRaises(AssertionError,   self.mesh.r, f, 'F', 'Fz', 'V')

    def test_mesh_r_E_M(self):
        g = np.ones((np.prod(self.mesh.vnEx), 2))
        g[:, 1] = 2
        Xex, Yex = self.mesh.r(g, 'Ex', 'Ex', 'M')
        self.assertTrue(np.all(Xex.shape == self.mesh.vnEx))
        self.assertTrue(np.all(Yex.shape == self.mesh.vnEx))
        self.assertTrue(np.all(Xex == 1))
        self.assertTrue(np.all(Yex == 2))

    def test_mesh_r_F_M(self):
        g = np.ones((np.prod(self.mesh.vnFx), 2))
        g[:, 1] = 2
        Xfx, Yfx = self.mesh.r(g, 'Fx', 'Fx', 'M')
        self.assertTrue(np.all(Xfx.shape == self.mesh.vnFx))
        self.assertTrue(np.all(Yfx.shape == self.mesh.vnFx))
        self.assertTrue(np.all(Xfx == 1))
        self.assertTrue(np.all(Yfx == 2))

    def test_mesh_r_CC_M(self):
        g = np.ones((self.mesh.nC, 2))
        g[:, 1] = 2
        Xc, Yc = self.mesh.r(g, 'CC', 'CC', 'M')
        self.assertTrue(np.all(Xc.shape == self.mesh.vnC))
        self.assertTrue(np.all(Yc.shape == self.mesh.vnC))
        self.assertTrue(np.all(Xc == 1))
        self.assertTrue(np.all(Yc == 2))

if __name__ == '__main__':
    unittest.main()
