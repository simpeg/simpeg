import numpy as np
from dask import array
from ....potential_fields.magnetics import Simulation3DIntegral as Sim
from ..base import BasePFSimulation
from ....utils import sdiag, mkvc


class Simulation3DIntegral(Sim, BasePFSimulation):
    """
    Overwrite the dask_getJtJdiag method
    """

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
                diag = array.einsum(
                    "i,ij,ij->j", W**2, self.Jmatrix, self.Jmatrix
                ).compute()
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
