from .simulation import (
    Simulation3DMagneticFluxDensity,
    Simulation3DElectricField,
    Simulation3DMagneticField,
    Simulation3DCurrentDensity,
)
from .simulation_1d import Simulation1DLayered
from .fields import (
    Fields3DMagneticFluxDensity,
    Fields3DElectricField,
    Fields3DMagneticField,
    Fields3DCurrentDensity,
)
from .survey import Survey
from . import sources
from . import receivers

from . import sources as Src
from . import receivers as Rx
