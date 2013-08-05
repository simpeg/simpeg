import numpy as np
import unittest
import sys
sys.path.append('../')
from OrderTest import OrderTest

MESHTYPES = ['uniformTensorMesh', 'uniformLOM']  # , 'rotateLOM'


class TestCurl(OrderTest):
    name = "Curl"
    meshTypes = MESHTYPES

    def getError(self):
        fun = lambda x: np.cos(x)  # i (cos(y)) + j (cos(z)) + k (cos(x))
        sol = lambda x: np.sin(x)  # i (sin(z)) + j (sin(x)) + k (sin(y))

        Ex = fun(self.M.gridEx[:, 1])
        Ey = fun(self.M.gridEy[:, 2])
        Ez = fun(self.M.gridEz[:, 0])
        E = np.concatenate((Ex, Ey, Ez))

        Fx = sol(self.M.gridFx[:, 2])
        Fy = sol(self.M.gridFy[:, 0])
        Fz = sol(self.M.gridFz[:, 1])
        curlE_anal = np.concatenate((Fx, Fy, Fz))

        # Generate DIV matrix
        CURL = self.M.edgeCurl

        curlE = CURL*E
        err = np.linalg.norm((curlE-curlE_anal), np.inf)
        return err

    def test_order(self):
        self.orderTest()


class TestFaceDiv(OrderTest):
    name = "Face Divergence"
    meshTypes = MESHTYPES

    def getError(self):
        DIV = self.M.faceDiv

        #Test function
        fun = lambda x: np.sin(x)
        Fx = fun(self.M.gridFx[:, 0])
        Fy = fun(self.M.gridFy[:, 1])
        Fz = fun(self.M.gridFz[:, 2])

        F = np.concatenate((Fx, Fy, Fz))
        divF = DIV*F
        sol = lambda x, y, z: (np.cos(x)+np.cos(y)+np.cos(z))
        divF_anal = sol(self.M.gridCC[:, 0], self.M.gridCC[:, 1], self.M.gridCC[:, 2])

        err = np.linalg.norm((divF-divF_anal), np.inf)

        return err

    def test_order(self):
        self.orderTest()


class TestFaceDiv2D(OrderTest):
    name = "Face Divergence 2D"
    meshTypes = MESHTYPES
    meshDimension = 2

    def getError(self):
        DIV = self.M.faceDiv

        #Test function
        fun = lambda x: np.sin(x)
        Fx = fun(self.M.gridFx[:, 0])
        Fy = fun(self.M.gridFy[:, 1])

        F = np.concatenate((Fx, Fy))
        divF = DIV*F
        sol = lambda x, y: (np.cos(x)+np.cos(y))
        divF_anal = sol(self.M.gridCC[:, 0], self.M.gridCC[:, 1])

        err = np.linalg.norm((divF-divF_anal), np.inf)

        return err

    def test_order(self):
        self.orderTest()


class TestNodalGrad(OrderTest):
    name = "Nodal Gradient"
    meshTypes = MESHTYPES

    def getError(self):
        GRAD = self.M.nodalGrad
        #Test function
        fun = lambda x, y, z: (np.cos(x)+np.cos(y)+np.cos(z))
        sol = lambda x: -np.sin(x)  # i (sin(x)) + j (sin(y)) + k (sin(z))

        phi = fun(self.M.gridN[:, 0], self.M.gridN[:, 1], self.M.gridN[:, 2])
        gradE = GRAD*phi

        Ex = sol(self.M.gridEx[:, 0])
        Ey = sol(self.M.gridEy[:, 1])
        Ez = sol(self.M.gridEz[:, 2])

        gradE_anal = np.concatenate((Ex, Ey, Ez))
        err = np.linalg.norm((gradE-gradE_anal), np.inf)

        return err

    def test_order(self):
        self.orderTest()


class TestNodalGrad2D(OrderTest):
    name = "Nodal Gradient 2D"
    meshTypes = MESHTYPES
    meshDimension = 2

    def getError(self):
        GRAD = self.M.nodalGrad
        #Test function
        fun = lambda x, y: (np.cos(x)+np.cos(y))
        sol = lambda x: -np.sin(x)  # i (sin(x)) + j (sin(y)) + k (sin(z))

        phi = fun(self.M.gridN[:, 0], self.M.gridN[:, 1])
        gradE = GRAD*phi

        Ex = sol(self.M.gridEx[:, 0])
        Ey = sol(self.M.gridEy[:, 1])

        gradE_anal = np.concatenate((Ex, Ey))
        err = np.linalg.norm((gradE-gradE_anal), np.inf)

        return err

    def test_order(self):
        self.orderTest()


if __name__ == '__main__':
    unittest.main()
