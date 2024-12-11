import numpy as np
from dask import array
from ....potential_fields.gravity import Simulation3DIntegral as Sim
from ...simulation import BaseSimulation
from ....utils import sdiag, mkvc


class Simulation3DIntegral(BaseSimulation, Sim):
    """
    Overload the Simulation3DIntegral class to use Dask
    """

    def getJtJdiag(self, m, W=None, f=None):
        """
        Return the diagonal of JtJ
        """

        self.model = m
        if W is None:
            W = np.ones(self.Jmatrix.shape[0])
        else:
            W = W.diagonal()

        if getattr(self, "_gtg_diagonal", None) is None:
            diag = array.einsum(
                "i,ij,ij->j", W**2, self.Jmatrix, self.Jmatrix
            ).compute()
            self._gtg_diagonal = diag
        else:
            diag = self._gtg_diagonal

        return mkvc((sdiag(np.sqrt(diag)) @ self.rhoDeriv).power(2).sum(axis=0))
