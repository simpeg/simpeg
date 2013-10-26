import numpy as np
import scipy.sparse.linalg as linalg


class Solver(object):
    """docstring for Solver"""
    def __init__(self, A, doDirect=True, flag=None, options={}):

        assert type(doDirect) is bool, 'doDirect must be a boolean'
        assert flag in [None, 'L', 'U', 'D'], "flag must be set to None, 'L', 'U', or 'D'"

        self.A = A

        self.dsolve = None
        self.doDirect = doDirect
        self.flag = flag
        self.options = options

    def solve(self, b):
        if self.flag is None and self.doDirect:
            return self.solveDirect(b, **self.options)
        elif self.flag is None and not self.doDirect:
            return self.solveIter(b, **self.options)
        elif self.flag == 'U':
            return self.solveBackward(b)
        elif self.flag == 'L':
            return self.solveForward(b)
        elif self.flag == 'D':
            return self.solveDiagonal(b)
        else:
            raise Exception('Unknown flag.')
        pass

    def clean(self):
        """Cleans up the memory"""
        del self.dsolve
        self.dsolve = None

    def solveDirect(self, b, factorize=False, backend='scipy'):
        assert np.shape(self.A)[1] == np.shape(b)[0], 'Dimension mismatch'

        if factorize and self.dsolve is None:
            self.A = self.A.tocsc()  # for efficiency
            self.dsolve = linalg.factorized(self.A)

        if len(b.shape) == 1 or b.shape[1] == 1:
            # Just one RHS
            if factorize:
                return self.dsolve(b)
            else:
                return linalg.dsolve.spsolve(self.A, b)

        # Multiple RHSs
        X = np.empty_like(b)
        for i in range(b.shape[1]):
            if factorize:
                X[:,i] = self.dsolve(b[:,i])
            else:
                X[:,i] = linalg.dsolve.spsolve(self.A,b[:,i])

        return X

    def solveIter(self, b, M=None, iterSolver='CG'):
        pass

    def solveBackward(self, b):
        pass

    def solveForward(self, b):
        pass

    def solveDiagonal(self, b):
        diagA = self.A.diagonal()
        if len(b.shape) == 1 or b.shape[1] == 1:
            # Just one RHS
            return b/diagA
        # Multiple RHSs
        X = np.empty_like(b)
        for i in range(b.shape[1]):
            X[:,i] = b[:,i]/diagA
        return X


if __name__ == '__main__':
    from SimPEG.mesh import TensorMesh
    from time import time
    h1 = np.ones(20)*100.
    h2 = np.ones(20)*100.
    h3 = np.ones(20)*100.

    h = [h1,h2,h3]

    M = TensorMesh(h)

    D = M.faceDiv
    G = M.cellGrad
    Msig = M.getFaceMass()
    A = D*Msig*G

    rhs = np.random.rand(M.nC)


    tic = time()
    solve = Solver(A, options={'factorize':True})
    x = solve.solve(rhs)
    print 'Factorized', time() - tic
    tic = time()
    solve = Solver(A, options={'factorize':False})
    x = solve.solve(rhs)
    print 'spsolve', time() - tic

