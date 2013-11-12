import unittest
from SimPEG import Solver
from SimPEG.mesh import TensorMesh
from SimPEG.utils import sdiag
import numpy as np
import scipy.sparse as sparse

TOL = 1e-10
numRHS = 5


class TestSolver(unittest.TestCase):

    def setUp(self):
        h1 = np.ones(10)*100.
        h2 = np.ones(10)*100.
        h3 = np.ones(10)*100.

        h = [h1,h2,h3]

        M = TensorMesh(h)

        D = M.faceDiv
        G = M.cellGrad
        Msig = M.getFaceMass()
        A = D*Msig*G
        A[0,0] *= 10 # remove the constant null space from the matrix

        self.A = A
        self.M = M

    def test_directFactored_1(self):
        solve = Solver(self.A, doDirect=True, flag=None, options={'factorize':True,'backend':'scipy'})
        e = np.ones(self.M.nC)
        rhs = self.A.dot(e)
        x = solve.solve(rhs)
        self.assertTrue(np.linalg.norm(e-x,np.inf) < TOL, True)


    def test_directFactored_M(self):
        solve = Solver(self.A, doDirect=True, flag=None, options={'factorize':True,'backend':'scipy'})
        e = np.ones((self.M.nC,numRHS))
        rhs = self.A.dot(e)
        x = solve.solve(rhs)
        self.assertTrue(np.linalg.norm(e-x,np.inf) < TOL, True)

    def test_directSpsolve_1(self):
        solve = Solver(self.A, doDirect=True, flag=None, options={'factorize':False,'backend':'scipy'})
        e = np.ones(self.M.nC)
        rhs = self.A.dot(e)
        x = solve.solve(rhs)
        self.assertTrue(np.linalg.norm(e-x,np.inf) < TOL, True)

    def test_directSpsolve_M(self):
        solve = Solver(self.A, doDirect=True, flag=None, options={'factorize':False,'backend':'scipy'})
        e = np.ones((self.M.nC, numRHS))
        rhs = self.A.dot(e)
        x = solve.solve(rhs)
        self.assertTrue(np.linalg.norm(e-x,np.inf) < TOL, True)

    def test_directLower_1_python(self):
        AL = sparse.tril(self.A)
        solve = Solver(AL, doDirect=True, flag='L', options={'backend':'python'})
        e = np.ones(self.M.nC)
        rhs = AL.dot(e)
        x = solve.solve(rhs)
        self.assertTrue(np.linalg.norm(e-x,np.inf) < TOL, True)

    def test_directLower_M_python(self):
        AL = sparse.tril(self.A)
        solve = Solver(AL, doDirect=True, flag='L', options={'backend':'python'})
        e = np.ones((self.M.nC,numRHS))
        rhs = AL.dot(e)
        x = solve.solve(rhs)

    def test_directLower_1_fortran(self):
        AL = sparse.tril(self.A)
        solve = Solver(AL, doDirect=True, flag='L', options={'backend':'fortran'})
        e = np.ones(self.M.nC)
        rhs = AL.dot(e)
        x = solve.solve(rhs)
        self.assertTrue(np.linalg.norm(e-x,np.inf) < TOL, True)

    def test_directLower_M_fortran(self):
        AL = sparse.tril(self.A)
        solve = Solver(AL, doDirect=True, flag='L', options={'backend':'fortran'})
        e = np.ones((self.M.nC,numRHS))
        rhs = AL.dot(e)
        x = solve.solve(rhs)
        self.assertTrue(np.linalg.norm(e-x,np.inf) < TOL, True)
        self.assertTrue(np.linalg.norm(e-x,np.inf) < TOL, True)

    def test_directUpper_1_python(self):
        AU = sparse.triu(self.A)
        solve = Solver(AU, doDirect=True, flag='U', options={})
        e = np.ones(self.M.nC)
        rhs = AU.dot(e)
        x = solve.solve(rhs)
        self.assertTrue(np.linalg.norm(e-x,np.inf) < TOL, True)

    def test_directUpper_M_python(self):
        AU = sparse.triu(self.A)
        solve = Solver(AU, doDirect=True, flag='U', options={})
        e = np.ones((self.M.nC,numRHS))
        rhs = AU.dot(e)
        x = solve.solve(rhs)
        self.assertTrue(np.linalg.norm(e-x,np.inf) < TOL, True)


    def test_directUpper_1_fortran(self):
        AU = sparse.triu(self.A)
        solve = Solver(AU, doDirect=True, flag='U', options={'backend':'fortran'})
        e = np.ones(self.M.nC)
        rhs = AU.dot(e)
        x = solve.solve(rhs)
        self.assertTrue(np.linalg.norm(e-x,np.inf) < TOL, True)

    def test_directUpper_M_fortran(self):
        AU = sparse.triu(self.A)
        solve = Solver(AU, doDirect=True, flag='U', options={'backend':'fortran'})
        e = np.ones((self.M.nC,numRHS))
        rhs = AU.dot(e)
        x = solve.solve(rhs)
        self.assertTrue(np.linalg.norm(e-x,np.inf) < TOL, True)

    def test_directDiagonal_1(self):
        AD = sdiag(self.A.diagonal())
        solve = Solver(AD, doDirect=True, flag='D', options={})
        e = np.ones(self.M.nC)
        rhs = AD.dot(e)
        x = solve.solve(rhs)
        self.assertTrue(np.linalg.norm(e-x,np.inf) < TOL, True)

    def test_directDiagonal_M(self):
        AD = sdiag(self.A.diagonal())
        solve = Solver(AD, doDirect=True, flag='D', options={})
        e = np.ones((self.M.nC,numRHS))
        rhs = AD.dot(e)
        x = solve.solve(rhs)
        self.assertTrue(np.linalg.norm(e-x,np.inf) < TOL, True)


if __name__ == '__main__':
    unittest.main()
