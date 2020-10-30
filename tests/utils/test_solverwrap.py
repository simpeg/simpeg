import unittest
from SimPEG.utils.solver_utils import Solver, SolverLU, SolverCG, SolverBiCG, SolverDiag
import scipy.sparse as sp
import numpy as np


class TestSolve(unittest.TestCase):
    def setUp(self):
        # Create a random matrix
        n = 400
        A = sp.random(n, n, density=0.25)

        self.n = n
        self.A = 0.5 * (A + A.T) + n * sp.eye(n)

    def test_Solver(self):
        x = np.random.rand(self.n)
        b = self.A @ x

        with self.assertWarns(Warning):
            Ainv = Solver(self.A, bad_kwarg=312)
        x2 = Ainv @ b
        np.testing.assert_almost_equal(x, x2)

    def test_SolverLU(self):
        x = np.random.rand(self.n)
        b = self.A @ x

        with self.assertWarns(Warning):
            Ainv = SolverLU(self.A, bad_kwarg=312)
        x2 = Ainv @ b
        np.testing.assert_almost_equal(x, x2)

    def test_SolverCG(self):
        x = np.random.rand(self.n)
        b = self.A @ x

        with self.assertWarns(Warning):
            Ainv = SolverCG(self.A, bad_kwarg=312)
        x2 = Ainv @ b
        np.testing.assert_almost_equal(x, x2, decimal=4)

    def test_SolverBiCG(self):
        x = np.random.rand(self.n)
        b = self.A @ x

        with self.assertWarns(Warning):
            Ainv = SolverBiCG(self.A, bad_kwarg=312)
        x2 = Ainv @ b
        np.testing.assert_almost_equal(x, x2, decimal=4)

    def test_SolverDiag(self):
        x = np.random.rand(self.n)
        A = sp.diags(np.random.randn(self.n))
        b = A @ x

        with self.assertWarns(Warning):
            Ainv = SolverDiag(A, bad_kwarg=312)
        x2 = Ainv @ b
        np.testing.assert_almost_equal(x, x2)


if __name__ == "__main__":
    unittest.main()
