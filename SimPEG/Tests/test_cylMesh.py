import unittest
import sys
from SimPEG import *


class TestCyl2DMesh(unittest.TestCase):

    def setUp(self):
        hx = np.r_[1,1,0.5]
        hz = np.r_[2,1]
        self.mesh = Mesh.CylMesh([hx, 1,hz])

    def test_cylMesh_numbers(self):
        self.assertTrue(self.mesh.dim == 3)

        self.assertTrue(self.mesh.nC == 6)
        self.assertTrue(self.mesh.nCx == 3)
        self.assertTrue(self.mesh.nCy == 1)
        self.assertTrue(self.mesh.nCz == 2)
        self.assertTrue(np.all(self.mesh.vnC == [3, 1, 2]))

        self.assertTrue(self.mesh.nN == 0)
        self.assertTrue(self.mesh.nNx == 3)
        self.assertTrue(self.mesh.nNy == 0)
        self.assertTrue(self.mesh.nNz == 3)
        self.assertTrue(np.all(self.mesh.vnN == [3, 0, 3]))

        self.assertTrue(self.mesh.nFx == 6)
        self.assertTrue(np.all(self.mesh.vnFx == [3, 1, 2]))
        self.assertTrue(self.mesh.nFy == 0)
        self.assertTrue(np.all(self.mesh.vnFy == [3, 0, 2]))
        self.assertTrue(self.mesh.nFz == 9)
        self.assertTrue(np.all(self.mesh.vnFz == [3, 1, 3]))
        self.assertTrue(self.mesh.nF == 15)
        self.assertTrue(np.all(self.mesh.vnF == [6, 0, 9]))

        self.assertTrue(self.mesh.nEx == 0)
        self.assertTrue(np.all(self.mesh.vnEx == [3, 0, 3]))
        self.assertTrue(self.mesh.nEy == 9)
        self.assertTrue(np.all(self.mesh.vnEy == [3, 1, 3]))
        self.assertTrue(self.mesh.nEz == 0)
        self.assertTrue(np.all(self.mesh.vnEz == [3, 0, 2]))
        self.assertTrue(self.mesh.nE == 9)
        self.assertTrue(np.all(self.mesh.vnE == [0, 9, 0]))

    def test_vectorsCC(self):
        v = np.r_[0, 1, 1.75]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorCCx)) == 0)
        v = np.r_[0]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorCCy)) == 0)
        v = np.r_[1, 2.5]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorCCz)) == 0)

    def test_vectorsN(self):
        v = np.r_[0.5, 1.5, 2]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorNx)) == 0)
        v = np.r_[np.pi] #This is kinda a fake. But it is where it would be if there was a radial connection
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorNy)) == 0)
        v = np.r_[0, 2, 3.]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorNz)) == 0)

    def test_dimensions(self):
        edge = np.r_[0.5, 1.5, 2, 0.5, 1.5, 2, 0.5, 1.5, 2] * 2 * np.pi
        self.assertTrue(np.linalg.norm((edge-self.mesh.edge)) == 0)

        r = np.r_[0, 0.5, 1.5, 2]
        a = r[1:]*2*np.pi
        areaX = np.r_[2*a,a]
        a = (r[1:]**2 - r[:-1]**2)*np.pi
        areaZ = np.r_[a,a,a]
        area = np.r_[areaX, areaZ]
        self.assertTrue(np.linalg.norm((area-self.mesh.area)) == 0)

        a = (r[1:]**2 - r[:-1]**2)*np.pi
        vol = np.r_[2*a,a]
        self.assertTrue(np.linalg.norm((vol-self.mesh.vol)) == 0)

    def test_faceDiv(self):
        print self.mesh.faceDiv

class TestCyl3DMesh(unittest.TestCase):

    def setUp(self):
        hx = np.r_[1,1,0.5]
        hy = np.r_[np.pi, np.pi]
        hz = np.r_[2,1]
        self.mesh = Mesh.CylMesh([hx, hy,hz])

    def test_cylMesh_numbers(self):
        self.assertTrue(self.mesh.nCx == 3)
        self.assertTrue(self.mesh.nCy == 2)
        self.assertTrue(self.mesh.nCz == 2)
        self.assertTrue(np.all(self.mesh.vnC == [3, 2, 2]))

        self.assertTrue(self.mesh.nN == 24)
        self.assertTrue(self.mesh.nNx == 4)
        self.assertTrue(self.mesh.nNy == 2)
        self.assertTrue(self.mesh.nNz == 3)
        self.assertTrue(np.all(self.mesh.vnN == [4, 2, 3]))

    def test_vectorsCC(self):
        v = np.r_[0.5, 1.5, 2.25]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorCCx)) == 0)
        v = np.r_[0, np.pi]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorCCy)) == 0)
        v = np.r_[1, 2.5]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorCCz)) == 0)

    def test_vectorsN(self):
        v = np.r_[0, 1, 2, 2.5]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorNx)) == 0)
        v = np.r_[np.pi/2, 1.5*np.pi]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorNy)) == 0)
        v = np.r_[0, 2, 3]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorNz)) == 0)


if __name__ == '__main__':
    unittest.main()
