from __future__ import print_function
import numpy as np
import unittest
from SimPEG.Mesh import TensorMesh
from SimPEG import Solver, Tests

TOL = 1e-10

class BasicTensorMeshTests(unittest.TestCase):

    def setUp(self):
        a = np.array([1, 1, 1])
        b = np.array([1, 2])
        c = np.array([1, 4])
        self.mesh2 = TensorMesh([a, b], [3, 5])
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

    def test_oneCell(self):
        hx = np.array([1e-5])
        M = TensorMesh([hx])
        self.assertTrue(M.nC == 1)

    def test_printing(self):
        print(TensorMesh([10]))
        print(TensorMesh([10,10]))
        print(TensorMesh([10,10,10]))

    def test_centering(self):
        M1d = TensorMesh([10], 'C')
        M2d = TensorMesh([10,10], 'CC')
        M3d = TensorMesh([10,10,10], 'CCC')
        self.assertLess(np.abs(M1d.x0 + 0.5).sum(), TOL)
        self.assertLess(np.abs(M2d.x0 + 0.5).sum(), TOL)
        self.assertLess(np.abs(M3d.x0 + 0.5).sum(), TOL)

    def test_negative(self):
        M1d = TensorMesh([10], 'N')
        self.assertRaises(Exception, TensorMesh, [10], 'F')
        M2d = TensorMesh([10,10], 'NN')
        M3d = TensorMesh([10,10,10], 'NNN')
        self.assertLess(np.abs(M1d.x0 + 1.0).sum(), TOL)
        self.assertLess(np.abs(M2d.x0 + 1.0).sum(), TOL)
        self.assertLess(np.abs(M3d.x0 + 1.0).sum(), TOL)

    def test_cent_neg(self):
        M3d = TensorMesh([10,10,10], 'C0N')
        self.assertLess(np.abs(M3d.x0 + np.r_[0.5,0,1.0]).sum(), TOL)

    def test_tensor(self):
        M = TensorMesh([[(10.,2)]])
        self.assertLess(np.abs(M.hx - np.r_[10.,10.]).sum(), TOL)

class TestPoissonEqn(Tests.OrderTest):
    name = "Poisson Equation"
    meshSizes = [10, 16, 20]

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
            fN = Solver(D*G) * (sol(self.M.gridCC))
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
