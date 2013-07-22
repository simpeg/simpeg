import numpy as np
from OrderTest import OrderTest
import unittest
from scipy.sparse.linalg import dsolve


class TestCurl(OrderTest):
    name = "Curl"

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


class TestNodalGrad(OrderTest):
    name = "Nodal Gradient"

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


class TestPoissonEqn(OrderTest):
    name = "Poisson Equation"
    meshSizes = [16, 20, 24]

    def getError(self):
        # Create some functions to integrate
        fun = lambda x: np.sin(2*np.pi*x[:, 0])*np.sin(2*np.pi*x[:, 1])*np.sin(2*np.pi*x[:, 2])
        sol = lambda x: -3.*((2*np.pi)**2)*fun(x)

        self.M.setCellGradBC('dirichlet')

        D = self.M.faceDiv
        G = self.M.cellGrad
        if self.forward:
            sA = sol(self.M.gridCC)
            sN = D*G*fun(self.M.gridCC)
            err = np.linalg.norm((sA - sN), np.inf)
        else:
            fA = fun(self.M.gridCC)
            fN = dsolve.spsolve(D*G, sol(self.M.gridCC))
            err = np.linalg.norm((fA - fN), np.inf)
        return err

    def test_orderForward(self):
        self.name = "Poisson Equation - Forward"
        self.forward = True
        self.orderTest()

    def test_orderBackward(self):
        self.name = "Poisson Equation - Backward"
        self.forward = False
        self.orderTest()

if __name__ == '__main__':
    unittest.main()
