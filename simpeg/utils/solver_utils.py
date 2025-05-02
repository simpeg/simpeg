import numpy as np
from scipy.sparse import linalg
from .mat_utils import mkvc
import warnings
from typing import Type

__all__ = [
    "Solver",
    "SolverLU",
    "SolverCG",
    "SolverBiCG",
    "Diagonal",
    "Pardiso",
    "Mumps",
    "wrap_direct",
    "wrap_iterative",
    "get_default_solver",
    "set_default_solver",
    "SolverWrapD",
    "SolverWrapI",
    "SolverDiag",
]

# The default direct solver priority is:
# Pardiso (optional, but available on intel systems)
# Mumps (optional, but available for all systems)
# Scipy's SuperLU (available for all scipy systems)
if AvailableSolvers["Pardiso"]:
    _DEFAULT_SOLVER = Pardiso
elif AvailableSolvers["Mumps"]:
    _DEFAULT_SOLVER = Mumps
else:
    _DEFAULT_SOLVER = SolverLU


# Create a specific warning allowing users to silence this if they so choose.
class DefaultSolverWarning(UserWarning):
    pass


def get_default_solver(warn=False) -> Type[Base]:
    """Return the default solver used by simpeg.

    Parameters
    ----------
    warn : bool, optional
        If True, a warning will be raised to let users know that the default
        solver is being chosen depending on their system.

    Returns
    -------
    Solver
        A new solver class created from a direct solver `fun`.

    Examples
    --------
    A solver that does not have a factorize method.

    >>> from simpeg.utils.solver_utils import SolverWrapD
    >>> import scipy.sparse as sp
    >>> SpSolver = SolverWrapD(sp.linalg.spsolve, factorize=False)
    >>> A = sp.diags([1, -1], [0, 1], shape=(10, 10))
    >>> b = np.arange(10)
    >>> Ainv = SpSolver(A)
    >>> x_solve = Ainv * b
    >>> A @ x_solve
    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

    Or one that has a factorize method (which can be re-used on multiple solves)

    >>> SolverLU = SolverWrapD(sp.linalg.splu, factorize=True)
    >>> A = sp.diags([1, -1], [0, 1], shape=(10, 10))
    >>> b = np.arange(10)
    >>> Ainv = SolverLU(A)
    >>> x_solve = Ainv * b
    >>> A @ x_solve
    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    """
    if warn:
        warnings.warn(
            f"Using the default solver: {_DEFAULT_SOLVER.__name__}. \n\n"
            f"If you would like to suppress this notification, add \n"
            f"warnings.filterwarnings("
            "'ignore', simpeg.utils.solver_utils.DefaultSolverWarning)\n"
            f" to your script.",
            DefaultSolverWarning,
            stacklevel=2,
        )
    return _DEFAULT_SOLVER


def SolverWrapI(fun, checkAccuracy=True, accuracyTol=1e-5, name=None):
    """Wraps an iterative Solver.

    Parameters
    ----------
    fun : function
        A function handle that accepts two arguments, a sparse matrix and a rhs array.
    checkAccuracy : bool, default: ``True``
        If ``True``, verify the accuracy of the solve
    accuracyTol : float, default: 1e-5
        Minimum accuracy of the solve
    name : str, optional
        A name for the function

    Returns
    -------
    Solver
        A new solver class created from the function.

    Examples
    --------

    >>> import scipy.sparse as sp
    >>> from simpeg.utils.solver_utils import SolverWrapI

    >>> SolverCG = SolverWrapI(sp.linalg.cg)
    >>> A = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(10, 10))
    >>> b = np.arange(10)
    >>> Ainv = SolverCG(A)
    >>> x_solve = Ainv * b
    >>> A @ x_solve
    array([3.55271368e-15, 1.00000000e+00, 2.00000000e+00, 3.00000000e+00,
       4.00000000e+00, 5.00000000e+00, 6.00000000e+00, 7.00000000e+00,
       8.00000000e+00, 9.00000000e+00])
    """

    def __init__(self, A, **kwargs):
        self.A = A

        self.checkAccuracy = kwargs.pop("checkAccuracy", checkAccuracy)
        self.accuracyTol = kwargs.pop("accuracyTol", accuracyTol)

        func_params = inspect.signature(fun).parameters
        # First test if function excepts **kwargs,
        # in which case we do not need to cull the kwargs
        do_cull = True
        for param_name in func_params:
            param = func_params[param_name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                do_cull = False
        if do_cull:
            # build a dictionary of valid kwargs
            culled_args = {}
            for item in kwargs:
                if item in func_params:
                    culled_args[item] = kwargs[item]
                else:
                    warnings.warn(
                        f"{item} is not a valid keyword for {fun.__name__} and will be ignored",
                        stacklevel=2,
                    )
            kwargs = culled_args

        self.kwargs = kwargs

    def __mul__(self, b):
        if not isinstance(b, np.ndarray):
            raise TypeError("Can only multiply by a numpy array.")

        if len(b.shape) == 1 or b.shape[1] == 1:
            b = b.flatten()
            # Just one RHS
            out = fun(self.A, b, **self.kwargs)
            if isinstance(out, tuple) and len(out) == 2:
                # We are dealing with scipy output with an info!
                X = out[0]
                self.info = out[1]
            else:
                X = out
        else:  # Multiple RHSs
            X = np.empty_like(b)
            for i in range(b.shape[1]):
                out = fun(self.A, b[:, i], **self.kwargs)
                if isinstance(out, tuple) and len(out) == 2:
                    # We are dealing with scipy output with an info!
                    X[:, i] = out[0]
                    self.info = out[1]
                else:
                    X[:, i] = out

        if self.checkAccuracy:
            _checkAccuracy(self.A, b, X, self.accuracyTol)
        return X

    def __matmul__(self, other):
        return self * other

    def clean(self):
        pass

    return type(
        name if name is not None else fun.__name__,
        (object,),
        {
            "__init__": __init__,
            "clean": clean,
            "__mul__": __mul__,
            "__matmul__": __matmul__,
        },
    )


Solver = SolverWrapD(linalg.spsolve, factorize=False, name="Solver")
SolverLU = SolverWrapD(linalg.splu, factorize=True, name="SolverLU")
SolverCG = SolverWrapI(linalg.cg, name="SolverCG")
SolverBiCG = SolverWrapI(linalg.bicgstab, name="SolverBiCG")


class SolverDiag(object):
    """Solver for a diagonal linear system

    This is a simple solver used for diagonal matrices.

    Parameters
    ----------
    A :
        A diagonal linear system

    Examples
    --------
    >>> import scipy.sparse as sp
    >>> from simpeg.utils.solver_utils import SolverDiag
    >>> A = sp.diags(np.linspace(1, 2, 10))
    >>> b = np.arange(10)
    >>> Ainv = SolverDiag(A)
    >>> x_solve = Ainv * b
    >>> A @ x_solve
    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    """

    def __init__(self, A, **kwargs):
        self.A = A
        self._diagonal = A.diagonal()
        for kwarg in kwargs:
            warnings.warn(
                f"{kwarg} is not recognized and will be ignored", stacklevel=2
            )

    def __mul__(self, rhs):
        n = self.A.shape[0]
        assert rhs.size % n == 0, "Incorrect shape of rhs."
        nrhs = rhs.size // n

        if len(rhs.shape) == 1 or rhs.shape[1] == 1:
            x = self._solve1(rhs)
        else:
            x = self._solveM(rhs)

        if nrhs == 1:
            return x.flatten()
        elif nrhs > 1:
            return x.reshape((n, nrhs), order="F")

    def __matmul__(self, other):
        return self * other

    def _solve1(self, rhs):
        return rhs.flatten() / self._diagonal

    def _solveM(self, rhs):
        n = self.A.shape[0]
        nrhs = rhs.size // n
        return rhs / self._diagonal.repeat(nrhs).reshape((n, nrhs))

    def clean(self):
        """Clean"""
        pass
