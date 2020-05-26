from .simulation import (
    Simulation3DMagneticFluxDensity,
    Simulation3DElectricField,
    Simulation3DMagneticField,
    Simulation3DCurrentDensity,
)
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

############
# Deprecated
############
from .simulation import Problem3D_e, Problem3D_b, Problem3D_h, Problem3D_j
from .fields import Fields3D_e, Fields3D_b, Fields3D_h, Fields3D_j
