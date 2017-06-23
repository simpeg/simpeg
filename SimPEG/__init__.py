from __future__ import print_function

from SimPEG import Utils
from SimPEG.Utils import mkvc
from SimPEG.Utils.SolverUtils import (
    _checkAccuracy, SolverWrapD, SolverWrapI,
    Solver, SolverCG, SolverDiag, SolverLU, SolverBiCG,
)

import discretize as Mesh
from SimPEG import Maps
from SimPEG import Models
from SimPEG import Problem
from SimPEG import Survey
from SimPEG import Regularization
from SimPEG import DataMisfit
from SimPEG import InvProblem
from SimPEG import Optimization
from SimPEG import Directives
from SimPEG import Inversion
from SimPEG import Tests

__version__   = '0.6.0'
__author__    = 'SimPEG Team'
__license__   = 'MIT'
__copyright__ = '2013 - 2017, SimPEG Team, http://simpeg.xyz'
