import numpy as np
from dask import array

from .....electromagnetics.static.induced_polarization.simulation import (
    BaseIPSimulation as Sim,
)
from SimPEG.dask.simulation import dask_Jvec, dask_Jtvec

Sim.Jvec = dask_Jvec
Sim.Jtvec = dask_Jtvec


def dask_getJtJdiag(self, W=None):
    """
    Return the diagonal of JtJ
    """
    if getattr(self, "_jtjdiag", None) is None:
        # Need to check if multiplying weights makes sense
        if W is None:
            W = self._scale ** 2.
        else:
            W = (self._scale * W.diagonal())**2.0

        diag = array.einsum('i,ij,ij->j', W, self.Jmatrix, self.Jmatrix)

        if isinstance(diag, array.Array):
            diag = np.asarray(diag.compute())

        self._jtjdiag = diag
    return self._jtjdiag


Sim.getJtJdiag = dask_getJtJdiag
