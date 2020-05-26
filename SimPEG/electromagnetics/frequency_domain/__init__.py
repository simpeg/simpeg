from .survey import Survey
from . import sources
from . import receivers
from .simulation import (
    Simulation3DElectricField,
    Simulation3DMagneticFluxDensity,
    Simulation3DCurrentDensity,
    Simulation3DMagneticField,
)
from .fields import (
    Fields3DElectricField,
    Fields3DMagneticFluxDensity,
    Fields3DCurrentDensity,
    Fields3DMagneticField,
)

from . import sources as Src
from . import receivers as Rx

############
# Deprecated
############
from .simulation import Problem3D_e, Problem3D_b, Problem3D_h, Problem3D_j
from .fields import Fields3D_e, Fields3D_b, Fields3D_h, Fields3D_j
