from . import survey
from . import sources
from . import receivers
from . import analytics
from . import simulation

from .simulation import Simulation3DIntegral, Simulation3DDifferential
from .survey import Survey
from .sources import SourceField
from .receivers import Point

############
# Deprecated
############
from ...utils.code_utils import deprecate_class
from .simulation import GravityIntegral, Problem3D_Diff
from .survey import LinearSurvey
from .receivers import RxObs
from .sources import SrcField

from ...maps import IdentityMap


@deprecate_class(removal_version="0.15.0", new_location="SimPEG.maps")
class BaseMagMap(IdentityMap):
    pass
