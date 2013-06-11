import unittest
import sys
sys.path.append('../')
from Mesh import Mesh
import numpy as np


class TestMeshNumbers3D(unittest.TestCase):

    def setUp(self):
        self.mesh = Mesh([6, 2, 3])

    def test_meshDimensions(self):
        self.assertTrue(self.mesh.dim, 3)

    def test_mesh_nc(self):
        self.assertTrue(np.all(self.mesh.nc == [6, 2, 3]))

    def test_mesh_nf(self):
        x = np.all(self.mesh.nfx == [7, 2, 3])
        y = np.all(self.mesh.nfy == [6, 3, 3])
        z = np.all(self.mesh.nfz == [6, 2, 4])

        self.assertTrue(np.all([x, y, z]))

    def test_mesh_ne(self):
        x = np.all(self.mesh.nex == [6, 3, 4])
        y = np.all(self.mesh.ney == [7, 2, 4])
        z = np.all(self.mesh.nez == [7, 3, 3])

        self.assertTrue(np.all([x, y, z]))

    def test_mesh_numbers(self):
        c = self.mesh.ncells == 36
        f = np.all(self.mesh.nfaces == [42, 54, 48])
        e = np.all(self.mesh.nedges == [72, 56, 63])

        self.assertTrue(np.all([c, f, e]))


class TestMeshNumbers2D(unittest.TestCase):

    def setUp(self):
        self.mesh = Mesh([6, 2])

    def test_meshDimensions(self):
        self.assertTrue(self.mesh.dim, 2)

    def test_mesh_nc(self):
        self.assertTrue(np.all(self.mesh.nc == [6, 2]))

    def test_mesh_nf(self):
        x = np.all(self.mesh.nfx == [7, 2])
        y = np.all(self.mesh.nfy == [6, 3])
        z = self.mesh.nfz is None

        self.assertTrue(np.all([x, y, z]))

    def test_mesh_ne(self):
        x = np.all(self.mesh.nex == [6, 3])
        y = np.all(self.mesh.ney == [7, 2])
        z = self.mesh.nez is None

        self.assertTrue(np.all([x, y, z]))

    def test_mesh_numbers(self):
        c = self.mesh.ncells == 12
        f = np.all(self.mesh.nfaces == [14, 18])
        e = np.all(self.mesh.nedges == [18, 14])

        self.assertTrue(np.all([c, f, e]))

if __name__ == '__main__':
    unittest.main()
