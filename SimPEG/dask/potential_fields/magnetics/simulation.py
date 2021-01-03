import numpy as np
from ....potential_fields.magnetics import Simulation3DIntegral as Sim
from ....utils import sdiag, mkvc
from dask import array, delayed
from scipy.sparse import csr_matrix as csr


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
        if not self.is_amplitude_data:
            diag = ((W[:, None] * self.Jmatrix) ** 2).sum(axis=0)
        else:  # self.modelType is amplitude
            fieldDeriv = self.fieldDeriv
            J = (
                fieldDeriv[0, :, None] * self.G[::3]
                + fieldDeriv[1, :, None] * self.G[1::3]
                + fieldDeriv[2, :, None] * self.G[2::3]
            )
            diag = ((W[:, None] * J) ** 2).sum(axis=0)
        self._gtg_diagonal = diag
    else:
        diag = self._gtg_diagonal

    return mkvc((sdiag(np.sqrt(diag)) @ self.chiDeriv).power(2).sum(axis=0))


Sim.getJtJdiag = dask_getJtJdiag


def dask_Jvec(self, _, v, f=None):
    """
    Sensitivity times a vector
    """
    dmu_dm_v = self.rhoDeriv @ v
    return array.dot(self.Jmatrix, dmu_dm_v.astype(np.float32))


Sim.Jvec = dask_Jvec


def dask_Jtvec(self, _, v, f=None):
    """
    Sensitivity transposed times a vector
    """

    Jtvec = array.dot(v.astype(np.float32), self.Jmatrix)
    Jtjvec_dmudm = delayed(csr.dot)(Jtvec, self.rhoDeriv)
    h_vec = array.from_delayed(
        Jtjvec_dmudm, dtype=float, shape=[self.rhoDeriv.shape[1]]
    )

    return h_vec


Sim.Jtvec = dask_Jtvec

