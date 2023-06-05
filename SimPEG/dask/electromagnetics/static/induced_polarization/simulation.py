import dask.array as da

from .....electromagnetics.static.induced_polarization.simulation import BaseIPSimulation as Sim


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
        w = da.from_array(W)[:, None]
        self._gtgdiag = da.sum((w * J) ** 2, axis=0).compute()

    return self._gtgdiag


Sim.getJtJdiag = dask_getJtJdiag
