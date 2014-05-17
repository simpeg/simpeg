import scipy.sparse as sp
from SimPEG.Utils import SolverUtils

Solver   = SolverUtils.DSolverWrap(sp.linalg.spsolve, factorize=False)
SolverLU = SolverUtils.DSolverWrap(sp.linalg.splu, factorize=True)
SolverCG = SolverUtils.ISolverWrap(sp.linalg.cg)
