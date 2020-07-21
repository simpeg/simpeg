from .simulation import Simulation3DCellCentered, Simulation3DNodal
from .simulation_2d import Simulation2DCellCentered, Simulation2DNodal
from .survey import Survey, from_dc_to_ip_survey
from .run import run_inversion
from ..resistivity import receivers
from ..resistivity import sources
from ..resistivity import utils

Simulation2DCellCentred = Simulation2DCellCentered
Simulation3DCellCentred = Simulation2DCellCentered

############
# Deprecated
############
from .simulation import Problem3D_CC, Problem3D_N
from .simulation_2d import Problem2D_CC, Problem2D_N
