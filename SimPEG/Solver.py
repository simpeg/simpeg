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

    def solveDirect(self, b, backend='scipy'):
        assert np.shape(self.A)[1] == np.shape(b)[0], 'Dimension mismatch'

        if self.dsolve is None:
            self.A = self.A.tocsc()  # for efficiency
            self.dsolve = linalg.factorized(self.A)

        if len(b.shape) == 1 or b.shape[1] == 1:
            # Just one RHS
            return self.dsolve(b)

        # Multiple RHSs
        X = np.empty_like(b)
        for i in range(b.shape[1]):
            X[:,i] = self.dsolve(b[:,i])

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
