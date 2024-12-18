import numpy as np
from dask import array, delayed
from ....potential_fields.gravity import Simulation3DIntegral as Sim
from ..base import G
from scipy.sparse import csr_matrix as csr


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
        diag = array.einsum("i,ij,ij->j", W**2, self.Jmatrix, self.Jmatrix)
        self._gtg_diagonal = diag
    else:
        diag = self._gtg_diagonal

    mapping_deriv = self.rhoDeriv.tocsr().T.power(2)
    dmudm_jtvec = delayed(csr.dot)(mapping_deriv, diag)
    jtjdiag = array.from_delayed(
        dmudm_jtvec, dtype=np.float32, shape=[mapping_deriv.shape[1]]
    )

    return jtjdiag


Sim.getJtJdiag = getJtJdiag
Sim.G = G
