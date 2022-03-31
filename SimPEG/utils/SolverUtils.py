from .code_utils import deprecate_module

deprecate_module("SolverUtils", "solver_utils", "0.16.0", error=True)

from .solver_utils import *
