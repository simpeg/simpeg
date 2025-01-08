from ....potential_fields.magnetics import Simulation3DIntegral as Sim
from ..base import G
from ...simulation import getJtJdiag


Sim.clean_on_model_update = []
Sim.getJtJdiag = getJtJdiag
Sim.G = G
