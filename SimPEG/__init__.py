import datetime as _datetime

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
__version__   = '0.10.0'
__author__    = 'SimPEG Team'
__license__   = 'MIT'
__copyright__ = f"2013 - {_datetime.datetime.now().year}, {__author__}, http://simpeg.xyz"
