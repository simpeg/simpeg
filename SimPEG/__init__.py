from __future__ import print_function
from __future__ import absolute_import

# import discretize as Mesh
import discretize
from discretize import Tests as tests

from .data import Data, SyntheticData
from . import data_misfit
from . import directives
from . import maps
from . import models
from . import inverse_problem
from . import inversion
from . import regularization
from . import survey
from . import simulation

from . import utils
from .utils import mkvc
from .utils import Report
from .utils.solver_utils import (
    _checkAccuracy,
    SolverWrapD,
    SolverWrapI,
    Solver,
    SolverCG,
    SolverDiag,
    SolverLU,
    SolverBiCG,
)

__version__ = "0.14.2"
__author__ = "SimPEG Team"
__license__ = "MIT"
__copyright__ = "2013 - 2020, SimPEG Team, http://simpeg.xyz"
