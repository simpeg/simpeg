""" module SimPEG.EM.NSEM

SimPEG implementation of the natural source problem
(including magenetotelluric, tipper and ZTEM)



"""

from . import Utils
from . import source as Src
from . import receiver as Rx
from .survey import Survey, Data
from .fields import Fields1D_ePrimSec, Fields3D_ePrimSec
from .simulation import Problem1D_ePrimSec, Problem3D_ePrimSec
