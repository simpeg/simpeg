from pymatsolver import (
    AvailableSolvers,
    Solver,
    SolverLU,
    SolverCG,
    SolverBiCG,
    Diagonal,
    Pardiso,
    Mumps,
    wrap_direct,
    wrap_iterative,
)
from pymatsolver.solvers import Base
from .code_utils import deprecate_function
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


def get_default_solver(warn=False) -> Type[Base]:
    """Return the default solver used by simpeg.

    Parameters
    ----------
    warn : bool, optional

        .. deprecated:: 0.25.0

           Argument ``warn`` is deprecated and will be removed in
           SimPEG v0.26.0.

    Returns
    -------
    solver
        The default solver class used by simpeg's simulations.
    """
    if warn:
        warnings.warn(
            "The `warn` argument has been deprecated and will be "
            "removed in SimPEG v0.26.0.",
            FutureWarning,
            stacklevel=2,
        )
    return _DEFAULT_SOLVER


def set_default_solver(solver_class: Type[Base]):
    """Set the default solver used by simpeg.

    Parameters
    ----------
    solver_class
        A ``pymatsolver.solvers.Base`` subclass used to construct an object
        that acts os the inverse of a sparse matrix.
    """
    global _DEFAULT_SOLVER
    if not issubclass(solver_class, Base):
        raise TypeError(
            "Default solver must be a subclass of pymatsolver.solvers.Base."
        )
    _DEFAULT_SOLVER = solver_class


# should likely deprecate these classes in favor of the pymatsolver versions.
SolverWrapD = deprecate_function(
    wrap_direct,
    old_name="SolverWrapD",
    removal_version="0.24.0",
    new_location="pymatsolver",
    error=True,
)
SolverWrapI = deprecate_function(
    wrap_iterative,
    old_name="SolverWrapI",
    removal_version="0.24.0",
    new_location="pymatsolver",
    error=True,
)
SolverDiag = deprecate_function(
    Diagonal,
    old_name="SolverDiag",
    removal_version="0.24.0",
    new_location="pymatsolver",
    error=True,
)
