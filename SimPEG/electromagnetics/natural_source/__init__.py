""" module SimPEG.EM.NSEM

SimPEG implementation of the natural source problem
(including magenetotelluric, tipper and ZTEM)



"""

from . import utils
from . import sources as Src
from . import receivers as Rx
from .survey import Survey, Data
from .fields import Fields1D_ePrimSec, Fields3D_ePrimSec
from .simulation import Problem1D_ePrimSec, Problem3D_ePrimSec
from . import sources
from . import receivers
