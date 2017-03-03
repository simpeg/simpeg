from __future__ import print_function

from SimPEG import Utils
from SimPEG.Utils import mkvc
from SimPEG.Utils.SolverUtils import (
    _checkAccuracy, SolverWrapD, SolverWrapI,
    Solver, SolverCG, SolverDiag, SolverLU, SolverBiCG,
)

import discretize as Mesh
from . import Maps
from . import Models
from . import Problem
from . import Survey
from . import Regularization
from . import DataMisfit
from . import InvProblem
from . import Optimization
from . import Directives
from . import Inversion
from . import Tests

__version__   = '0.5.0'
__author__    = 'Rowan Cockett'
__license__   = 'MIT'
__copyright__ = 'Copyright 2014 Rowan Cockett'
