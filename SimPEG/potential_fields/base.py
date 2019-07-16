from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import properties
from scipy.constants import mu_0
import numpy as np

from ..data import Data
from ..maps import IdentityMap
from ..simulation import BaseSimulation
from ..survey import BaseSurvey, BaseSrc
from ..utils import sdiag, Zero
from .. import props

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

__all__ = ['BaseEMSimulation', 'BaseEMSurvey', 'BaseEMSrc']



###############################################################################
#                                                                             #
#                             Base Potential Fields Problem                   #
#                                                                             #
###############################################################################

class BasePFSimulation(LinearSimulation):

    chi, chiMap, chiDeriv = Props.Invertible(
        "Magnetic Susceptibility (SI)",
        default=1.
    )

    rho, rhoMap, rhoDeriv = Props.Invertible(
        "Specific density (g/cc)",
        default=1.
    )
    # mapPair = IdentityMap  #: Type of mapping to pair with

    solver = Solver  #: Type of solver to pair with

    verbose = False

