import unittest
import sys
from SimPEG import *
from TestUtils import OrderTest


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
        v = np.r_[0, 1.5, 2.25]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorCCx)) == 0)
        v = np.r_[0]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorCCy)) == 0)
        v = np.r_[1, 2.5]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorCCz)) == 0)

    def test_vectorsN(self):
        v = np.r_[1, 2, 2.5]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorNx)) == 0)
        v = np.r_[0]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorNy)) == 0)
        v = np.r_[0, 2, 3.]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorNz)) == 0)

    def test_dimensions(self):
        edge = np.r_[1, 2, 2.5, 1, 2, 2.5, 1, 2, 2.5] * 2 * np.pi
        self.assertTrue(np.linalg.norm((edge-self.mesh.edge)) == 0)

        r = np.r_[0, 1, 2, 2.5]
        a = r[1:]*2*np.pi
        areaX = np.r_[2*a,a]
        a = (r[1:]**2 - r[:-1]**2)*np.pi
        areaZ = np.r_[a,a,a]
        area = np.r_[areaX, areaZ]
        self.assertTrue(np.linalg.norm((area-self.mesh.area)) == 0)

        a = (r[1:]**2 - r[:-1]**2)*np.pi
        vol = np.r_[2*a,a]
        self.assertTrue(np.linalg.norm((vol-self.mesh.vol)) == 0)

    def test_gridSizes(self):
        self.assertTrue(self.mesh.gridCC.shape == (self.mesh.nC, 3))
        # self.assertTrue(self.mesh.gridN.shape == (self.mesh.nN, 3))

        self.assertTrue(self.mesh.gridFx.shape == (self.mesh.nFx, 3))
        # self.assertTrue(self.mesh.gridFy.shape == (self.mesh.nFy, 3))
        self.assertTrue(self.mesh.gridFz.shape == (self.mesh.nFz, 3))

        # self.assertTrue(self.mesh.gridEx.shape == (self.mesh.nEx, 3))
        self.assertTrue(self.mesh.gridEy.shape == (self.mesh.nEy, 3))
        # self.assertTrue(self.mesh.gridEz.shape == (self.mesh.nEz, 3))


MESHTYPES = ['uniformCylMesh']
MESHDIMENSION = 2
call2 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1])
call3 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])
cart_row2 = lambda g, xfun, yfun: np.c_[call2(xfun, g), call2(yfun, g)]
cart_row3 = lambda g, xfun, yfun, zfun: np.c_[call3(xfun, g), call3(yfun, g), call3(zfun, g)]
cartF2 = lambda M, fx, fy: np.vstack((cart_row2(M.gridFx, fx, fy), cart_row2(M.gridFy, fx, fy)))
cartE2 = lambda M, ex, ey: np.vstack((cart_row2(M.gridEx, ex, ey), cart_row2(M.gridEy, ex, ey)))
cartF3 = lambda M, fx, fy, fz: np.vstack((cart_row3(M.gridFx, fx, fy, fz), cart_row3(M.gridFy, fx, fy, fz), cart_row3(M.gridFz, fx, fy, fz)))
cartE3 = lambda M, ex, ey, ez: np.vstack((cart_row3(M.gridEx, ex, ey, ez), cart_row3(M.gridEy, ex, ey, ez), cart_row3(M.gridEz, ex, ey, ez)))


class TestFaceDiv(OrderTest):
    name = "FaceDiv"
    meshTypes = MESHTYPES
    meshDimension = MESHDIMENSION

    def getError(self):

        funX = lambda x, y, z: np.cos(2*np.pi*y)
        funY = lambda x, y, z: 1+x*0
        funZ = lambda x, y, z: np.cos(2*np.pi*x)

        solX = lambda x, y, z: 2*np.pi*np.sin(2*np.pi*z)
        solY = lambda x, y, z: 2*np.pi*np.sin(2*np.pi*x)
        solZ = lambda x, y, z: 2*np.pi*np.sin(2*np.pi*y)

        Ec = cartE3(self.M, funX, funY, funZ)
        print Ec.shape, self.M.nE
        print self.M
        E = self.M.projectEdgeVector(Ec)

        Fc = cartF3(self.M, solX, solY, solZ)
        curlE_anal = self.M.projectFaceVector(Fc)

        curlE = self.M.edgeCurl.dot(E)
        err = np.linalg.norm((curlE - curlE_anal), np.inf)
        return err

    # def test_order(self):
    #     self.orderTest()

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
