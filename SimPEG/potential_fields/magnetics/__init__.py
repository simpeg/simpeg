from . import survey
from . import sources
from . import receivers
from . import analytics
from . import simulation

from .simulation import Simulation3DIntegral, Simulation3DDifferential
from .survey import MagneticSurvey
from .sources import SourceField
from .receivers import point_receiver

############
# Deprecated
############
from .simulation import MagneticIntegral, Problem3D_Diff
