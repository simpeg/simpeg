import dask.array as da
import dask
import numpy as np
from ....potential_fields.magnetics import Simulation3DIntegral as Sim
from ....utils import sdiag

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

    if self.modelType != 'amplitude':
        diag = ((W[:, None]*self.G)**2).sum(axis=0).compute()
    else:  # self.modelType is amplitude
        fieldDeriv = self.fieldDeriv
        J = (fieldDeriv[0, :, None] * self.G[::3] +
             fieldDeriv[1, :, None] * self.G[1::3] +
             fieldDeriv[2, :, None] * self.G[2::3])
        diag = ((W[:, None]*J)**2).sum(axis=0).compute()
    self._gtg_diagonal = ((sdiag(np.sqrt(diag))@self.chiDeriv)**2).sum(axis=0)
    return self._gtg_diagonal
Sim.getJtJdiag = dask_getJtJdiag
