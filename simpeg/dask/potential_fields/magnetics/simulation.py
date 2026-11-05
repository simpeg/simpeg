import numpy as np
from ....potential_fields.magnetics import Simulation3DIntegral as Sim
from ....utils import sdiag, mkvc


def dask_getJtJdiag(self, m, W=None, f=None):
    """
    Return the diagonal of JtJ
    """

    self.model = m

    self.model = m
    if W is None:
        W = np.ones(self.Jmatrix.shape[0])
    else:
        W = W.diagonal()

    if getattr(self, "_gtg_diagonal", None) is None:
        if not self.is_amplitude_data:
            diag = np.asarray(np.einsum("i,ij,ij->j", W**2, self.Jmatrix, self.Jmatrix))
        else:
            ampDeriv = self.ampDeriv
            J = (
                ampDeriv[0, :, None] * self.Jmatrix[::3]
                + ampDeriv[1, :, None] * self.Jmatrix[1::3]
                + ampDeriv[2, :, None] * self.Jmatrix[2::3]
            )
            diag = ((W[:, None] * J) ** 2).sum(axis=0).compute()
        self._gtg_diagonal = diag
    else:
        diag = self._gtg_diagonal

    return mkvc((sdiag(np.sqrt(diag)) @ self.chiDeriv).power(2).sum(axis=0))


Sim.getJtJdiag = dask_getJtJdiag


@property
def G(self):
    """
    Gravity forward operator
    """
    if getattr(self, "_G", None) is None:
        self._G = self.Jmatrix

    return self._G


Sim._delete_on_model_update = []
Sim.G = G
