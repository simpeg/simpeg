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
from . import utils
from .IODC import IO
from .run import run_inversion

Simulation2DCellCentred = Simulation2DCellCentered
Simulation3DCellCentred = Simulation3DCellCentered
Fields3DCellCentred = Fields3DCellCentered
Fields2DCellCentred = Fields2DCellCentered
