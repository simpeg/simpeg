import numpy as np
from matutils import mkvc
import warnings

def DSolverWrap(fun, factorize=True, destroy = False, checkAccuracy=True, accuracyTol=1e-6):

    def __init__(self, A, **kwargs):

        self.A = A.tocsc()
        self.kwargs = kwargs
        if factorize:
            self.solver = fun(self.A, **kwargs)

    def solve(self, b):
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
            nrm = np.linalg.norm(mkvc(self.A*X - b)) / np.linalg.norm(mkvc(b))
            if nrm > accuracyTol:
                msg = '### SolverWarning ###: Accuracy on solve is above tolerance: %e > %e' % (nrm, accuracyTol)
                print msg
                warnings.warn(msg, RuntimeWarning)
        return X

    def clean(self):
        if destroy == True:
            return self.solver.clean()
        else:
            return True

    return type(fun.__name__, (object,), {"__init__": __init__, "solve": solve, "clean": clean})



def ISolverWrap(fun, checkAccuracy=True, accuracyTol=1e-5):

    def __init__(self, A, **kwargs):
        self.A = A.tocsc()
        self.kwargs = kwargs

    def solve(self, b):
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
            nrm = np.linalg.norm(mkvc(self.A*X - b)) / np.linalg.norm(mkvc(b))
            if nrm > accuracyTol:
                msg = '### SolverWarning ###: Accuracy on solve is above tolerance: %e > %e' % (nrm, accuracyTol)
                print msg
                warnings.warn(msg, RuntimeWarning)
        return X

    return type(fun.__name__, (object,), {"__init__": __init__, "solve": solve})
