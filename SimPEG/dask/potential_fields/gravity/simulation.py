import numpy as np
from ....potential_fields.gravity import Simulation3DIntegral as Sim
from ....utils import sdiag, mkvc


def dask_getJtJdiag(self, m, W=None):
    """
        Return the diagonal of JtJ
    """

    self.model = m

    if W is None:
        W = np.ones(self.nD)
    else:
        W = W.diagonal()
    if getattr(self, "_gtg_diagonal", None) is None:
        diag = ((W[:, None] * self.G) ** 2).sum(axis=0).compute()
        self._gtg_diagonal = diag
    else:
        diag = self._gtg_diagonal
    return mkvc((sdiag(np.sqrt(diag)) @ self.rhoDeriv).power(2).sum(axis=0))


Sim.getJtJdiag = dask_getJtJdiag
