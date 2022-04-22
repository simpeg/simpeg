""" module SimPEG.electromagnetics.natural_source

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
from .simulation_1d import Simulation1DRecursive
