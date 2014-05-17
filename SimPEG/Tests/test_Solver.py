import unittest
from SimPEG import *
from SimPEG.Mesh import TensorMesh
from SimPEG.Utils import sdiag
import numpy as np
import scipy.sparse as sparse

TOL = 1e-10
numRHS = 5

def dotest(solver, multi=False, **solverOpts):
    h1 = np.ones(10)*100.
    h2 = np.ones(10)*100.
    h3 = np.ones(10)*100.

    h = [h1,h2,h3]

    M = TensorMesh(h)

    D = M.faceDiv
    G = -M.faceDiv.T
    Msig = M.getFaceInnerProduct()
    A = D*Msig*G
    A[0,0] *= 10 # remove the constant null space from the matrix


    Ainv = Solver(A, **solverOpts)
    if multi:
        e = np.ones(M.nC)
    else:
        e = np.ones((M.nC, numRHS))
    rhs = A * e
    x = Ainv * rhs
    Ainv.clean()
    return np.linalg.norm(e-x,np.inf) < TOL

class TestSolver(unittest.TestCase):

    def test_direct_spsolve_1(self): self.assertTrue(dotest(Solver, False))
    def test_direct_spsolve_M(self): self.assertTrue(dotest(Solver, True))

    def test_direct_splu_1(self): self.assertTrue(dotest(SolverLU, False))
    def test_direct_splu_M(self): self.assertTrue(dotest(SolverLU, True))

    def test_iterative_cg_1(self): self.assertTrue(dotest(SolverCG, False))
    def test_iterative_cg_M(self): self.assertTrue(dotest(SolverCG, True))


if __name__ == '__main__':
    unittest.main()
