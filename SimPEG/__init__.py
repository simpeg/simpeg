from __future__ import print_function
from __future__ import absolute_import

import discretize as Mesh
from discretize import Tests

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

from . import Utils
from .Utils import mkvc
from .Utils import versions
from .Utils.SolverUtils import (
    _checkAccuracy, SolverWrapD, SolverWrapI,
    Solver, SolverCG, SolverDiag, SolverLU, SolverBiCG,
)
__version__   = '0.9.1'
__author__    = 'SimPEG Team'
__license__   = 'MIT'
__copyright__ = '2013 - 2018, SimPEG Team, http://simpeg.xyz'
