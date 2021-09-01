import numpy as np
from ....potential_fields.gravity import Simulation3DIntegral as Sim
from ....utils import sdiag, mkvc
from dask import array, delayed
from scipy.sparse import csr_matrix as csr
from dask.distributed import Future, get_client


def dask_fields(self, m):
    """
    Fields computed from a linear operation
    """
    self.model = m

    if self.store_sensitivities == "forward_only":
        self.model = m
        # Compute the linear operation without forming the full dense G
        fields = self.linear_operator()
    else:
        fields = self.G @ (self.rhoMap @ m).astype(np.float32)

    return fields


Sim.fields = dask_fields


def dask_getJtJdiag(self, m, W=None):
    """
    Return the diagonal of JtJ
    """

    self.model = m

    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    if W is None:
        W = np.ones(self.nD)
    else:
        W = W.diagonal()

    if getattr(self, "_gtg_diagonal", None) is None:
        diag = ((W[:, None] * self.Jmatrix) ** 2).sum(axis=0).compute()
        self._gtg_diagonal = diag
    else:
        diag = self._gtg_diagonal
    return mkvc((sdiag(np.sqrt(diag)) @ self.rhoDeriv).power(2).sum(axis=0))


Sim.getJtJdiag = dask_getJtJdiag


def dask_Jvec(self, _, v, f=None):
    """
    Sensitivity times a vector
    """

    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    dmu_dm_v = self.rhoDeriv @ v
    return array.dot(self.Jmatrix, dmu_dm_v.astype(np.float32))


Sim.Jvec = dask_Jvec


def dask_Jtvec(self, _, v, f=None):
    """
    Sensitivity transposed times a vector
    """

    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    Jtvec = array.dot(v.astype(np.float32), self.Jmatrix)
    Jtjvec_dmudm = delayed(csr.dot)(Jtvec, self.rhoDeriv)
    h_vec = array.from_delayed(
        Jtjvec_dmudm, dtype=float, shape=[self.rhoDeriv.shape[1]]
    )

    return h_vec


Sim.Jtvec = dask_Jtvec


@property
def dask_G(self):
    """
    The linear forward operator
    """
    return self.Jmatrix


Sim.G = dask_G
