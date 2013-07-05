import unittest
import sys
sys.path.append('../')
from BaseMesh import BaseMesh
import numpy as np


class TestMeshNumbers3D(unittest.TestCase):

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
        f = np.all(self.mesh.nF == [42, 54, 48])
        e = np.all(self.mesh.nE == [72, 56, 63])

        self.assertTrue(np.all([c, f, e]))


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
        f = np.all(self.mesh.nF == [14, 18])
        e = np.all(self.mesh.nE == [18, 14])

        self.assertTrue(np.all([c, f, e]))

if __name__ == '__main__':
    unittest.main()
