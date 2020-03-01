""" module SimPEG.EM.NSEM

SimPEG implementation of the natural source problem
(including magenetotelluric, tipper and ZTEM)



"""

from . import utils
from . import sources as Src
from . import receivers as Rx
from .survey import Survey, Data
from .fields import Fields1DPrimarySecondary, Fields3DPrimarySecondary
from .simulation import Simulation1DPrimarySecondary, Simulation3DPrimarySecondary
from . import sources
from . import receivers

############
# Deprecated
############
from .simulation import Problem3D_ePrimSec, Problem1D_ePrimSec
