import numpy as np, scipy.sparse as sp
from matutils import mkvc
import warnings

def _checkAccuracy(A, b, X, accuracyTol):
    nrm = np.linalg.norm(mkvc(A*X - b), np.inf)
    nrm_b = np.linalg.norm(mkvc(b), np.inf)
    if nrm_b > 0:
        nrm /= nrm_b
    if nrm > accuracyTol:
        msg = '### SolverWarning ###: Accuracy on solve is above tolerance: %e > %e' % (nrm, accuracyTol)
        print msg
        warnings.warn(msg, RuntimeWarning)


def SolverWrapD(fun, factorize=True, checkAccuracy=True, accuracyTol=1e-6):
    """
    Wraps a direct Solver.

    ::

        Solver   = SolverUtils.SolverWrapD(sp.linalg.spsolve, factorize=False)
        SolverLU = SolverUtils.SolverWrapD(sp.linalg.splu, factorize=True)

    """

    def __init__(self, A, **kwargs):
        self.A = A.tocsc()
        self.kwargs = kwargs
        if factorize:
            self.solver = fun(self.A, **kwargs)

    def __mul__(self, b):
        if type(b) is not np.ndarray:
            raise TypeError('Can only multiply by a numpy array.')

        if len(b.shape) == 1 or b.shape[1] == 1:
            b = b.flatten()
            # Just one RHS
            if factorize:
                X = self.solver.solve(b, **self.kwargs)
            else:
                X = fun(self.A, b, **self.kwargs)
        else: # Multiple RHSs
            X = np.empty_like(b)
            for i in range(b.shape[1]):
                if factorize:
                    X[:,i] = self.solver.solve(b[:,i])
                else:
                    X[:,i] = fun(self.A, b[:,i], **self.kwargs)

        if checkAccuracy:
            _checkAccuracy(self.A, b, X, accuracyTol)
        return X

    def clean(self):
        if factorize and hasattr(self.solver, 'clean'):
            return self.solver.clean()

    return type(fun.__name__+'_Wrapped', (object,), {"__init__": __init__, "clean": clean, "__mul__": __mul__})



def SolverWrapI(fun, checkAccuracy=True, accuracyTol=1e-5):
    """
    Wraps an iterative Solver.

    ::

        SolverCG = SolverUtils.SolverWrapI(sp.linalg.cg)

    """

    def __init__(self, A, **kwargs):
        self.A = A
        self.kwargs = kwargs

    def __mul__(self, b):
        if type(b) is not np.ndarray:
            raise TypeError('Can only multiply by a numpy array.')

        if len(b.shape) == 1 or b.shape[1] == 1:
            b = b.flatten()
            # Just one RHS
            out = fun(self.A, b, **self.kwargs)
            if type(out) is tuple and len(out) == 2:
                # We are dealing with scipy output with an info!
                X = out[0]
                self.info = out[1]
            else:
                X = out
        else: # Multiple RHSs
            X = np.empty_like(b)
            for i in range(b.shape[1]):
                out = fun(self.A, b[:,i], **self.kwargs)
                if type(out) is tuple and len(out) == 2:
                    # We are dealing with scipy output with an info!
                    X[:,i] = out[0]
                    self.info = out[1]
                else:
                    X[:,i] = out

        if checkAccuracy:
            _checkAccuracy(self.A, b, X, accuracyTol)
        return X

    def clean(self):
        pass

    return type(fun.__name__+'_Wrapped', (object,), {"__init__": __init__, "clean": clean, "__mul__": __mul__})


Solver   = SolverWrapD(sp.linalg.spsolve, factorize=False)
SolverLU = SolverWrapD(sp.linalg.splu, factorize=True)
SolverCG = SolverWrapI(sp.linalg.cg)


class SolverDiag(object):
    """docstring for SolverDiag"""
    def __init__(self, A):
        self.A = A
        self._diagonal = A.diagonal()

    def __mul__(self, rhs):
        n = self.A.shape[0]
        assert rhs.size % n == 0, 'Incorrect shape of rhs.'
        nrhs = rhs.size // n

        if len(rhs.shape) == 1 or rhs.shape[1] == 1:
            x = self._solve1(rhs)
        else:
            x = self._solveM(rhs)

        if nrhs == 1:
            return x.flatten()
        elif nrhs > 1:
            return x.reshape((n,nrhs), order='F')

    def _solve1(self, rhs):
        return rhs.flatten()/self._diagonal

    def _solveM(self, rhs):
        n = self.A.shape[0]
        nrhs = rhs.size // n
        return rhs/self._diagonal.repeat(nrhs).reshape((n,nrhs))

    def clean(self):
        pass
