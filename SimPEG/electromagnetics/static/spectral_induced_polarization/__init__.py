from ....data import Data
from .simulation import Simulation3DCellCentered, Simulation3DNodal
from .simulation_2d import Simulation2DCellCentered, Simulation2DNodal
from .survey import Survey, from_dc_to_sip_survey
from . import sources
from . import receivers
from . import sources as Src
from . import receivers as Rx
from .run import run_inversion, spectral_ip_mappings


Simulation2DCellCentred = Simulation2DCellCentered
Simulation3DCellCentred = Simulation2DCellCentered

############
# Deprecated
############
from .simulation import Problem3D_CC, Problem3D_N
from .simulation_2d import Problem2D_CC, Problem2D_N
