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
from .simulation import MagneticIntegral, Problem3D_Diff
from .survey import LinearSurvey
from .receivers import RxObs
from .sources import SrcField

from ...maps import ChiMap, Weighting


@deprecate_class(removal_version="0.15.0", new_location="SimPEG.maps")
class BaseMagMap(ChiMap):
    pass


@deprecate_class(removal_version="0.15.0", new_location="SimPEG.maps")
class WeightMap(Weighting):
    pass
