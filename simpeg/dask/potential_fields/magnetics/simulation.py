import numpy as np
from dask import array
from ....potential_fields.magnetics import Simulation3DIntegral as Sim
from ..base import G
from ....utils import sdiag, mkvc


def getJtJdiag(self, m, W=None, f=None):
    """
    Return the diagonal of JtJ
    """

    self.model = m

    if W is None:
        W = np.ones(self.nD)
    else:
        W = W.diagonal()
    if getattr(self, "_gtg_diagonal", None) is None:
        if not self.is_amplitude_data:
            diag = array.einsum("i,ij,ij->j", W**2, self.Jmatrix, self.Jmatrix)
        else:
            ampDeriv = self.ampDeriv
            J = (
                ampDeriv[0, :, None] * self.Jmatrix[::3]
                + ampDeriv[1, :, None] * self.Jmatrix[1::3]
                + ampDeriv[2, :, None] * self.Jmatrix[2::3]
            )
            diag = array.einsum("i,ij,ij->j", W**2, J, J)
        self._gtg_diagonal = np.asarray(diag)

    return mkvc(
        (sdiag(np.sqrt(self._gtg_diagonal)) @ self.chiDeriv).power(2).sum(axis=0)
    )


Sim.clean_on_model_update = []
Sim.getJtJdiag = getJtJdiag
Sim.G = G
