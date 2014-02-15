import unittest
import sys
from SimPEG import *


class TestCylMesh(unittest.TestCase):

    def setUp(self):
        hx = np.ones(10)
        hz = np.ones(5)

        self.mesh = Mesh.CylMesh([hx, 1,hz])

    def test_cylMeshInheritance(self):
        self.assertTrue(isinstance(self.mesh, Mesh.BaseMesh))

    def test_cylMeshDimensions(self):
        self.assertTrue(self.mesh.dim == 3)

    def test_cylMesh_nc(self):
        self.assertTrue(np.all(self.mesh.vnC == [10, 1, 5]))

    def test_cylMesh_nc_xyz(self):
        self.assertTrue(self.mesh.nCx == 10)
        self.assertTrue(self.mesh.nCy == 1)
        self.assertTrue(self.mesh.nCz == 5)

if __name__ == '__main__':
    unittest.main()
