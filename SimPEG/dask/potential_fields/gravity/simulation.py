import dask.array as da
import dask
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
    if getattr(self, "_gtg_diagonal", None) is not None:
        return self._gtg_diagonal

    diag = ((W[:, None]*self.G)**2).sum(axis=0).compute()

    self._gtg_diagonal = mkvc(
        ((sdiag(np.sqrt(diag))@self.rhoDeriv).power(2)).sum(axis=0)
    )
    return self._gtg_diagonal
Sim.getJtJdiag = dask_getJtJdiag
