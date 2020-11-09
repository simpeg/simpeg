# Copyright (c) 2013 SimPEG Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the SimPEG project (https://simpeg.xyz)
from discretize import Tests
import discretize as Mesh

from . import maps as Maps
from . import models as Models
from . import simulation as Problem
from . import survey as Survey
from . import regularization as Regularization
from . import data_misfit as DataMisfit
from . import inverse_problem as InvProblem
from . import optimization as Optimization
from . import directives as Directives
from . import inversion as Inversion

from .utils import mkvc
from .utils import versions
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
