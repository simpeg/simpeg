from __future__ import print_function
from __future__ import absolute_import

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

from SimPEG import Utils
from SimPEG.Utils import mkvc
from SimPEG.Utils.SolverUtils import (
    _checkAccuracy, SolverWrapD, SolverWrapI,
    Solver, SolverCG, SolverDiag, SolverLU, SolverBiCG,
)

__version__   = '0.5.0'
__author__    = 'Rowan Cockett'
__license__   = 'MIT'
__copyright__ = 'Copyright 2014 Rowan Cockett'
