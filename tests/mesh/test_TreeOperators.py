import numpy as np
import unittest
from SimPEG.Tests import OrderTest
import matplotlib.pyplot as plt

#TODO: 'randomTensorMesh'
MESHTYPES = ['uniformTree'] #['randomTree', 'uniformTree']
call2 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1])
call3 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])
cart_row2 = lambda g, xfun, yfun: np.c_[call2(xfun, g), call2(yfun, g)]
cart_row3 = lambda g, xfun, yfun, zfun: np.c_[call3(xfun, g), call3(yfun, g), call3(zfun, g)]
cartF2 = lambda M, fx, fy: np.vstack((cart_row2(M.gridFx, fx, fy), cart_row2(M.gridFy, fx, fy)))
cartE2 = lambda M, ex, ey: np.vstack((cart_row2(M.gridEx, ex, ey), cart_row2(M.gridEy, ex, ey)))
cartF3 = lambda M, fx, fy, fz: np.vstack((cart_row3(M.gridFx, fx, fy, fz), cart_row3(M.gridFy, fx, fy, fz), cart_row3(M.gridFz, fx, fy, fz)))
cartE3 = lambda M, ex, ey, ez: np.vstack((cart_row3(M.gridEx, ex, ey, ez), cart_row3(M.gridEy, ex, ey, ez), cart_row3(M.gridEz, ex, ey, ez)))


class TestFaceDiv2D(OrderTest):
    name = "Face Divergence 2D"
    meshTypes = MESHTYPES
    meshDimension = 2
    meshSizes = [16, 32]

    def getError(self):
        #Test function
        fx = lambda x, y: np.sin(2*np.pi*x)
        fy = lambda x, y: np.sin(2*np.pi*y)
        sol = lambda x, y: 2*np.pi*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))

        Fc = cartF2(self.M, fx, fy)
        F = self.M.projectFaceVector(Fc)

        divF = self.M.faceDiv.dot(F)
        divF_ana = call2(sol, self.M.gridCC)

        err = np.linalg.norm((divF-divF_ana), np.inf)

        # self.M.plotImage(divF-divF_ana, showIt=True)

        return err

    def test_order(self):
        self.orderTest()

class TestFaceDiv3D(OrderTest):
    name = "Face Divergence 3D"
    meshTypes = MESHTYPES
    meshSizes = [8, 16]

    def getError(self):
        #Test function
        fx = lambda x, y, z: np.sin(2*np.pi*x)
        fy = lambda x, y, z: np.sin(2*np.pi*y)
        fz = lambda x, y, z: np.sin(2*np.pi*z)
        sol = lambda x, y, z: (2*np.pi*np.cos(2*np.pi*x)+2*np.pi*np.cos(2*np.pi*y)+2*np.pi*np.cos(2*np.pi*z))

        Fc = cartF3(self.M, fx, fy, fz)
        F = self.M.projectFaceVector(Fc)

        divF = self.M.faceDiv.dot(F)
        divF_ana = call3(sol, self.M.gridCC)

        return np.linalg.norm((divF-divF_ana), np.inf)


    def test_order(self):
        self.orderTest()


class TestCurl(OrderTest):
    name = "Curl"
    meshTypes = MESHTYPES
    meshSizes = [4, 8, 16, 32]

    def getError(self):
        # fun: i (cos(y)) + j (cos(z)) + k (cos(x))
        # sol: i (sin(z)) + j (sin(x)) + k (sin(y))

        funX = lambda x, y, z: np.cos(2*np.pi*y)
        funY = lambda x, y, z: np.cos(2*np.pi*z)
        funZ = lambda x, y, z: np.cos(2*np.pi*x)

        solX = lambda x, y, z: 2*np.pi*np.sin(2*np.pi*z)
        solY = lambda x, y, z: 2*np.pi*np.sin(2*np.pi*x)
        solZ = lambda x, y, z: 2*np.pi*np.sin(2*np.pi*y)

        Ec = cartE3(self.M, funX, funY, funZ)
        E = self.M.projectEdgeVector(Ec)

        Fc = cartF3(self.M, solX, solY, solZ)
        curlE_ana = self.M.projectFaceVector(Fc)

        curlE = self.M.edgeCurl.dot(E)

        err = np.linalg.norm((curlE - curlE_ana), np.inf)

        return err

    def test_order(self):
        self.orderTest()

if __name__ == '__main__':
    unittest.main()
