import numpy as np
import unittest
import sys
sys.path.append('../')
from OrderTest import OrderTest

MESHTYPES = ['uniformTensorMesh', 'uniformLOM', 'rotateLOM']
call2 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1])
call3 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])
cart_row2 = lambda g, xfun, yfun: np.c_[call2(xfun, g), call2(yfun, g)]
cart_row3 = lambda g, xfun, yfun, zfun: np.c_[call3(xfun, g), call3(yfun, g), call3(zfun, g)]
cartF2 = lambda M, fx, fy: np.vstack((cart_row2(M.gridFx, fx, fy), cart_row2(M.gridFy, fx, fy)))
cartE2 = lambda M, ex, ey: np.vstack((cart_row2(M.gridEx, ex, ey), cart_row2(M.gridEy, ex, ey)))
cartF3 = lambda M, fx, fy, fz: np.vstack((cart_row3(M.gridFx, fx, fy, fz), cart_row3(M.gridFy, fx, fy, fz), cart_row3(M.gridFz, fx, fy, fz)))
cartE3 = lambda M, ex, ey, ez: np.vstack((cart_row3(M.gridEx, ex, ey, ez), cart_row3(M.gridEy, ex, ey, ez), cart_row3(M.gridEz, ex, ey, ez)))


class TestCurl(OrderTest):
    name = "Curl"
    meshTypes = MESHTYPES

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
        curlE_anal = self.M.projectFaceVector(Fc)

        curlE = self.M.edgeCurl.dot(E)
        if self._meshType == 'rotateLOM':
            # Really it is the integration we should be caring about:
            # So, let us look at the l2 norm.
            err = np.linalg.norm(self.M.area*(curlE - curlE_anal), 2)
        else:
            err = np.linalg.norm((curlE - curlE_anal), np.inf)
        return err

    def test_order(self):
        self.orderTest()


class TestFaceDiv(OrderTest):
    name = "Face Divergence"
    meshTypes = MESHTYPES
    meshSizes = [8, 16, 32]

    def getError(self):
        #Test function
        fx = lambda x, y, z: np.sin(2*np.pi*x)
        fy = lambda x, y, z: np.sin(2*np.pi*y)
        fz = lambda x, y, z: np.sin(2*np.pi*z)
        sol = lambda x, y, z: (2*np.pi*np.cos(2*np.pi*x)+2*np.pi*np.cos(2*np.pi*y)+2*np.pi*np.cos(2*np.pi*z))

        Fc = cartF3(self.M, fx, fy, fz)
        F = self.M.projectFaceVector(Fc)

        divF = self.M.faceDiv.dot(F)
        divF_anal = call3(sol, self.M.gridCC)

        if self._meshType == 'rotateLOM':
            # Really it is the integration we should be caring about:
            # So, let us look at the l2 norm.
            err = np.linalg.norm(self.M.vol*(divF-divF_anal), 2)
        else:
            err = np.linalg.norm((divF-divF_anal), np.inf)
        return err

    def test_order(self):
        self.orderTest()


class TestFaceDiv2D(OrderTest):
    name = "Face Divergence 2D"
    meshTypes = MESHTYPES
    meshDimension = 2
    meshSizes = [8, 16, 32, 64]

    def getError(self):
        #Test function
        fx = lambda x, y: np.sin(2*np.pi*x)
        fy = lambda x, y: np.sin(2*np.pi*y)
        sol = lambda x, y: 2*np.pi*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))

        Fc = cartF2(self.M, fx, fy)
        F = self.M.projectFaceVector(Fc)

        divF = self.M.faceDiv.dot(F)
        divF_anal = call2(sol, self.M.gridCC)

        err = np.linalg.norm((divF-divF_anal), np.inf)

        return err

    def test_order(self):
        self.orderTest()


class TestNodalGrad(OrderTest):
    name = "Nodal Gradient"
    meshTypes = MESHTYPES

    def getError(self):
        #Test function
        fun = lambda x, y, z: (np.cos(x)+np.cos(y)+np.cos(z))
        # i (sin(x)) + j (sin(y)) + k (sin(z))
        solX = lambda x, y, z: -np.sin(x)
        solY = lambda x, y, z: -np.sin(y)
        solZ = lambda x, y, z: -np.sin(z)

        phi = call3(fun, self.M.gridN)
        gradE = self.M.nodalGrad.dot(phi)

        Ec = cartE3(self.M, solX, solY, solZ)
        gradE_anal = self.M.projectEdgeVector(Ec)

        err = np.linalg.norm((gradE-gradE_anal), np.inf)

        return err

    def test_order(self):
        self.orderTest()


class TestNodalGrad2D(OrderTest):
    name = "Nodal Gradient 2D"
    meshTypes = MESHTYPES
    meshDimension = 2

    def getError(self):
        #Test function
        fun = lambda x, y: (np.cos(x)+np.cos(y))
        # i (sin(x)) + j (sin(y)) + k (sin(z))
        solX = lambda x, y: -np.sin(x)
        solY = lambda x, y: -np.sin(y)

        phi = call2(fun, self.M.gridN)
        gradE = self.M.nodalGrad.dot(phi)

        Ec = cartE2(self.M, solX, solY)
        gradE_anal = self.M.projectEdgeVector(Ec)

        err = np.linalg.norm((gradE-gradE_anal), np.inf)

        return err

    def test_order(self):
        self.orderTest()


if __name__ == '__main__':
    unittest.main()
