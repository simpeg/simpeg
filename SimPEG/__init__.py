from __future__ import print_function
from __future__ import absolute_import

# import discretize as Mesh
from discretize import Tests as tests

# from . import Maps
# from . import Models
# from . import Problem
# from . import Survey
# from . import Regularization
# from . import DataMisfit
from . import InvProblem
from . import Optimization
from . import Directives
from . import Inversion
# from . import Tests

from . import data
from . import data_misfit
from . import maps
from . import models
from . import regularization
from . import survey
from . import simulation


from . import utils
from .utils import mkvc
from .utils import versions
from .utils.SolverUtils import (
    _checkAccuracy, SolverWrapD, SolverWrapI,
    Solver, SolverCG, SolverDiag, SolverLU, SolverBiCG,
)
__version__   = '0.11.4'
__author__    = 'SimPEG Team'
__license__   = 'MIT'
__copyright__ = '2013 - 2019, SimPEG Team, http://simpeg.xyz'
