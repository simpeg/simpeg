from pymatsolver import (
    AvailableSolvers,
    Solver,
    SolverLU,
    SolverCG,
    SolverBiCG,
    Diagonal,
    WrapDirect,
    WrapIterative,
)
from .code_utils import deprecate_function
import scipy.sparse as sp
import numpy as np
import warnings

__all__ = [
    "Solver",
    "SolverLU",
    "SolverCG",
    "SolverBiCG",
    "Diagonal",
    "WrapDirect",
    "WrapIterative",
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
    from pymatsolver import Pardiso

    __all__ += ["Pardiso"]

    _DEFAULT_SOLVER = Pardiso
elif AvailableSolvers["Mumps"]:
    from pymatsolver import Mumps

    __all__ += ["Mumps"]

    _DEFAULT_SOLVER = Mumps
else:
    _DEFAULT_SOLVER = SolverLU


# Create a specific warning allowing users to silence this if they so choose.
class DefaultSolverWarning(UserWarning):
    pass


def get_default_solver():
    """Return the default solver used by simpeg.

    Returns
    -------
    solver class
    """
    warnings.warn(
        f"Using the default solver: {_DEFAULT_SOLVER.__name__}",
        DefaultSolverWarning,
        stacklevel=2,
    )
    return _DEFAULT_SOLVER


def set_default_solver(solver):
    """Set the default solver used by simpeg.

    Parameters
    ----------
    solver
        a solver like class used to construct an object
        that acts os the inverse of a sparse matrix, and
        supports the `__mul__` operation (at a miniumum).
    """
    global _DEFAULT_SOLVER
    # likely need to do some simple tests to make sure that the class is a valid
    # solver class?
    # test that you can pass it a scipy sparse matrix
    try:
        A = sp.csr_matrix(
            [
                [
                    0.5,
                    0.0,
                ],
                [0.0, 2.0],
            ]
        )
        Ainv = solver(A)
    except TypeError as err:
        raise TypeError(
            "Invalid solver type, unable to pass a csr_matrix to the constructor."
        ) from err

    # test that you can multiply a single right hand side to the solver
    try:
        b = np.array([1.0, -1.0])
        _ = Ainv * b
    except Exception as err:
        raise TypeError(
            "Unable to multiply the solver instance by a single numpy vector"
        ) from err

    # test that you can multiply it by a 2D column vector.
    try:
        b = np.array([[1.0], [-1.0]])
        _ = Ainv * b
    except Exception as err:
        raise TypeError(
            "Unable to multiply the solver instance by multiple numpy vectors"
        ) from err

    _DEFAULT_SOLVER = solver


# should likely deprecate these classes in favor of the pymatsolver versions.
SolverWrapD = deprecate_function(
    WrapDirect,
    old_name="SolverWrapD",
    removal_version="0.24.0",
    new_name="pymatsolver.WrapDirect",
)
SolverWrapI = deprecate_function(
    WrapIterative,
    old_name="SolverWrapI",
    removal_version="0.24.0",
    new_name="pymatsolver.WrapIterative",
)
SolverDiag = deprecate_function(
    Diagonal,
    old_name="SolverDiag",
    removal_version="0.24.0",
    new_name="pymatsolver.Diagonal",
)
