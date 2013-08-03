import numpy as np
import unittest
import sys
sys.path.append('../')
from TensorMesh import TensorMesh
from OrderTest import OrderTest
from scipy.sparse.linalg import dsolve


class BasicTensorMeshTests(unittest.TestCase):

    def setUp(self):
        a = np.array([1, 1, 1])
        b = np.array([1, 2])
        c = np.array([1, 4])
        self.mesh2 = TensorMesh([a, b], np.array([3, 5]))
        self.mesh3 = TensorMesh([a, b, c])

    def test_vectorN_2D(self):
        testNx = np.array([3, 4, 5, 6])
        testNy = np.array([5, 6, 8])

        xtest = np.all(self.mesh2.vectorNx == testNx)
        ytest = np.all(self.mesh2.vectorNy == testNy)
        self.assertTrue(xtest and ytest)

    def test_vectorCC_2D(self):
        testNx = np.array([3.5, 4.5, 5.5])
        testNy = np.array([5.5, 7])

        xtest = np.all(self.mesh2.vectorCCx == testNx)
        ytest = np.all(self.mesh2.vectorCCy == testNy)
        self.assertTrue(xtest and ytest)

    def test_area_3D(self):
        test_area = np.array([1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2])
        t1 = np.all(self.mesh3.area == test_area)
        self.assertTrue(t1)

    def test_vol_3D(self):
        test_vol = np.array([1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8])
        t1 = np.all(self.mesh3.vol == test_vol)
        self.assertTrue(t1)

    def test_vol_2D(self):
        test_vol = np.array([1, 1, 1, 2, 2, 2])
        t1 = np.all(self.mesh2.vol == test_vol)
        self.assertTrue(t1)

    def test_edge_3D(self):
        test_edge = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        t1 = np.all(self.mesh3.edge == test_edge)
        self.assertTrue(t1)

    def test_edge_2D(self):
        test_edge = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
        t1 = np.all(self.mesh2.edge == test_edge)
        self.assertTrue(t1)


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
