import dask.array as da

from .....electromagnetics.static.induced_polarization.simulation import (
    BaseIPSimulation as Sim,
)


def dask_getJtJdiag(self, m, W=None, f=None):
    """
    Return the diagonal of JtJ
    """
    if getattr(self, "_gtgdiag", None) is None:
        J = self.getJ(m, f=f)
        # Need to check if multiplying weights makes sense
        if W is None:
            W = self._scale
        else:
            W = self._scale * W.diagonal()
        self._gtgdiag = da.einsum("i,ij,ij->j", W**2, J, J).compute()

    return self._gtgdiag


Sim.getJtJdiag = dask_getJtJdiag
