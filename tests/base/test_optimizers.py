from __future__ import print_function
import unittest
from SimPEG import Solver
from discretize import TensorMesh
from SimPEG.utils import sdiag
import numpy as np
import scipy.sparse as sp
from SimPEG import optimization
from discretize.Tests import getQuadratic, Rosenbrock

TOL = 1e-2


class TestOptimizers(unittest.TestCase):
    def setUp(self):
        self.A = sp.identity(2).tocsr()
        self.b = np.array([-5, -5])

    def test_GN_Rosenbrock(self):
        GN = optimization.GaussNewton()
        xopt = GN.minimize(Rosenbrock, np.array([0, 0]))
        x_true = np.array([1.0, 1.0])
        print("xopt: ", xopt)
        print("x_true: ", x_true)
        self.assertTrue(np.linalg.norm(xopt - x_true, 2) < TOL, True)

    def test_GN_quadratic(self):
        GN = optimization.GaussNewton()
        xopt = GN.minimize(getQuadratic(self.A, self.b), np.array([0, 0]))
        x_true = np.array([5.0, 5.0])
        print("xopt: ", xopt)
        print("x_true: ", x_true)
        self.assertTrue(np.linalg.norm(xopt - x_true, 2) < TOL, True)

    def test_ProjGradient_quadraticBounded(self):
        PG = optimization.ProjectedGradient(debug=True)
        PG.lower, PG.upper = -2, 2
        xopt = PG.minimize(getQuadratic(self.A, self.b), np.array([0, 0]))
        x_true = np.array([2.0, 2.0])
        print("xopt: ", xopt)
        print("x_true: ", x_true)
        self.assertTrue(np.linalg.norm(xopt - x_true, 2) < TOL, True)

    def test_ProjGradient_quadratic1Bound(self):
        myB = np.array([-5, 1])
        PG = optimization.ProjectedGradient()
        PG.lower, PG.upper = -2, 2
        xopt = PG.minimize(getQuadratic(self.A, myB), np.array([0, 0]))
        x_true = np.array([2.0, -1.0])
        print("xopt: ", xopt)
        print("x_true: ", x_true)
        self.assertTrue(np.linalg.norm(xopt - x_true, 2) < TOL, True)

    def test_NewtonRoot(self):
        fun = (
            lambda x, return_g=True: np.sin(x)
            if not return_g
            else (np.sin(x), sdiag(np.cos(x)))
        )
        x = np.array([np.pi - 0.3, np.pi + 0.1, 0])
        xopt = optimization.NewtonRoot(comments=False).root(fun, x)
        x_true = np.array([np.pi, np.pi, 0])
        print("Newton Root Finding")
        print("xopt: ", xopt)
        print("x_true: ", x_true)
        self.assertTrue(np.linalg.norm(xopt - x_true, 2) < TOL, True)


if __name__ == "__main__":
    unittest.main()
