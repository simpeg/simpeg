from ....potential_fields.gravity import Simulation3DIntegral as Sim
from ...simulation import getJtJdiag


@property
def G(self):
    """
    Gravity forward operator
    """
    if getattr(self, "_G", None) is None:
        self._G = self.Jmatrix

    return self._G


Sim._delete_on_model_update = []
Sim.getJtJdiag = getJtJdiag
Sim.G = G
