from .simulation import Simulation3DCellCentered, Simulation3DNodal
from .simulation_2d import Simulation2DCellCentered, Simulation2DNodal
from .simulation_1d import Simulation1DLayers
from .survey import Survey, Survey_ky
from . import sources
from . import receivers
from . import sources as Src
from . import receivers as Rx
from .fields import FieldsDC, Fields3DCellCentered, Fields3DNodal
from .fields_2d import Fields2D, Fields2DCellCentered, Fields2DNodal
from .boundary_utils import getxBCyBC_CC
from . import utils
from .IODC import IO
from .run import run_inversion

Simulation2DCellCentred = Simulation2DCellCentered
Simulation3DCellCentred = Simulation2DCellCentered
Fields3DCellCentred = Fields3DCellCentered
Fields2DCellCentred = Fields2DCellCentered

############
# Deprecated
############
from .simulation import Problem3D_CC, Problem3D_N
from .simulation_2d import Problem2D_CC, Problem2D_N
from .fields import Fields_CC, Fields_N
from .fields_2d import Fields_ky, Fields_ky_CC, Fields_ky_N
