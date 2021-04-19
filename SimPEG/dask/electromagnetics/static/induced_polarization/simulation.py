from .....electromagnetics.static.induced_polarization.simulation import (
    BaseIPSimulation as Sim,
)

import dask.array as da


def dask_getJtJdiag(self, m, W=None):
    """
        Return the diagonal of JtJ
    """
    if self.gtgdiag is None:

        # Need to check if multiplying weights makes sense
        if W is None:
            W = self._scale
        else:
            W = self._scale * W.diagonal()
        w = da.from_array(W)[:, None]
        self.gtgdiag = da.sum((w * self.getJ(m)) ** 2, axis=0).compute()

    return self.gtgdiag


Sim.getJtJdiag = dask_getJtJdiag
