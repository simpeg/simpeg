import unittest
import sys
from SimPEG.Mesh import BaseMesh
import numpy as np


class TestBaseMesh(unittest.TestCase):

    def setUp(self):
        self.mesh = BaseMesh([6, 2, 3])

    def test_meshDimensions(self):
        self.assertTrue(self.mesh.dim, 3)

    def test_mesh_nc(self):
        self.assertTrue(np.all(self.mesh.n == [6, 2, 3]))

    def test_mesh_nc_xyz(self):
        x = np.all(self.mesh.nCx == 6)
        y = np.all(self.mesh.nCy == 2)
        z = np.all(self.mesh.nCz == 3)

        self.assertTrue(np.all([x, y, z]))

    def test_mesh_nf(self):
        x = np.all(self.mesh.nFx == [7, 2, 3])
        y = np.all(self.mesh.nFy == [6, 3, 3])
        z = np.all(self.mesh.nFz == [6, 2, 4])

        self.assertTrue(np.all([x, y, z]))

    def test_mesh_ne(self):
        x = np.all(self.mesh.nEx == [6, 3, 4])
        y = np.all(self.mesh.nEy == [7, 2, 4])
        z = np.all(self.mesh.nEz == [7, 3, 3])

        self.assertTrue(np.all([x, y, z]))

    def test_mesh_numbers(self):
        c = self.mesh.nC == 36
        fv = np.all(self.mesh.nFv == [42, 54, 48])
        ev = np.all(self.mesh.nEv == [72, 56, 63])
        f = np.all(self.mesh.nF == np.sum([42, 54, 48]))
        e = np.all(self.mesh.nE == np.sum([72, 56, 63]))

        self.assertTrue(np.all([c, fv, ev, f, e]))

    def test_mesh_r_E_V(self):
        ex = np.ones(self.mesh.nEv[0])
        ey = np.ones(self.mesh.nEv[1])*2
        ez = np.ones(self.mesh.nEv[2])*3
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
        fx = np.ones(self.mesh.nFv[0])
        fy = np.ones(self.mesh.nFv[1])*2
        fz = np.ones(self.mesh.nFv[2])*3
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
        g = np.ones((np.prod(self.mesh.nEx), 3))
        g[:, 1] = 2
        g[:, 2] = 3
        Xex, Yex, Zex = self.mesh.r(g, 'Ex', 'Ex', 'M')
        self.assertTrue(np.all(Xex.shape == self.mesh.nEx))
        self.assertTrue(np.all(Yex.shape == self.mesh.nEx))
        self.assertTrue(np.all(Zex.shape == self.mesh.nEx))
        self.assertTrue(np.all(Xex == 1))
        self.assertTrue(np.all(Yex == 2))
        self.assertTrue(np.all(Zex == 3))

    def test_mesh_r_F_M(self):
        g = np.ones((np.prod(self.mesh.nFx), 3))
        g[:, 1] = 2
        g[:, 2] = 3
        Xfx, Yfx, Zfx = self.mesh.r(g, 'Fx', 'Fx', 'M')
        self.assertTrue(np.all(Xfx.shape == self.mesh.nFx))
        self.assertTrue(np.all(Yfx.shape == self.mesh.nFx))
        self.assertTrue(np.all(Zfx.shape == self.mesh.nFx))
        self.assertTrue(np.all(Xfx == 1))
        self.assertTrue(np.all(Yfx == 2))
        self.assertTrue(np.all(Zfx == 3))

    def test_mesh_r_CC_M(self):
        g = np.ones((self.mesh.nC, 3))
        g[:, 1] = 2
        g[:, 2] = 3
        Xc, Yc, Zc = self.mesh.r(g, 'CC', 'CC', 'M')
        self.assertTrue(np.all(Xc.shape == self.mesh.n))
        self.assertTrue(np.all(Yc.shape == self.mesh.n))
        self.assertTrue(np.all(Zc.shape == self.mesh.n))
        self.assertTrue(np.all(Xc == 1))
        self.assertTrue(np.all(Yc == 2))
        self.assertTrue(np.all(Zc == 3))


class TestMeshNumbers2D(unittest.TestCase):

    def setUp(self):
        self.mesh = BaseMesh([6, 2])

    def test_meshDimensions(self):
        self.assertTrue(self.mesh.dim, 2)

    def test_mesh_nc(self):
        self.assertTrue(np.all(self.mesh.n == [6, 2]))

    def test_mesh_nc_xyz(self):
        x = np.all(self.mesh.nCx == 6)
        y = np.all(self.mesh.nCy == 2)
        z = self.mesh.nCz is None

        self.assertTrue(np.all([x, y, z]))

    def test_mesh_nf(self):
        x = np.all(self.mesh.nFx == [7, 2])
        y = np.all(self.mesh.nFy == [6, 3])
        z = self.mesh.nFz is None

        self.assertTrue(np.all([x, y, z]))

    def test_mesh_ne(self):
        x = np.all(self.mesh.nEx == [6, 3])
        y = np.all(self.mesh.nEy == [7, 2])
        z = self.mesh.nEz is None

        self.assertTrue(np.all([x, y, z]))

    def test_mesh_numbers(self):
        c = self.mesh.nC == 12
        fv = np.all(self.mesh.nFv == [14, 18])
        ev = np.all(self.mesh.nEv == [18, 14])
        f = np.all(self.mesh.nF == np.sum([14, 18]))
        e = np.all(self.mesh.nE == np.sum([18, 14]))

        self.assertTrue(np.all([c, fv, ev, f, e]))

    def test_mesh_r_E_V(self):
        ex = np.ones(self.mesh.nEv[0])
        ey = np.ones(self.mesh.nEv[1])*2
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
        fx = np.ones(self.mesh.nFv[0])
        fy = np.ones(self.mesh.nFv[1])*2
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
        g = np.ones((np.prod(self.mesh.nEx), 2))
        g[:, 1] = 2
        Xex, Yex = self.mesh.r(g, 'Ex', 'Ex', 'M')
        self.assertTrue(np.all(Xex.shape == self.mesh.nEx))
        self.assertTrue(np.all(Yex.shape == self.mesh.nEx))
        self.assertTrue(np.all(Xex == 1))
        self.assertTrue(np.all(Yex == 2))

    def test_mesh_r_F_M(self):
        g = np.ones((np.prod(self.mesh.nFx), 2))
        g[:, 1] = 2
        Xfx, Yfx = self.mesh.r(g, 'Fx', 'Fx', 'M')
        self.assertTrue(np.all(Xfx.shape == self.mesh.nFx))
        self.assertTrue(np.all(Yfx.shape == self.mesh.nFx))
        self.assertTrue(np.all(Xfx == 1))
        self.assertTrue(np.all(Yfx == 2))

    def test_mesh_r_CC_M(self):
        g = np.ones((self.mesh.nC, 2))
        g[:, 1] = 2
        Xc, Yc = self.mesh.r(g, 'CC', 'CC', 'M')
        self.assertTrue(np.all(Xc.shape == self.mesh.n))
        self.assertTrue(np.all(Yc.shape == self.mesh.n))
        self.assertTrue(np.all(Xc == 1))
        self.assertTrue(np.all(Yc == 2))

if __name__ == '__main__':
    unittest.main()
