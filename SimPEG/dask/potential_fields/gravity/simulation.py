import numpy as np
from ....potential_fields.gravity import Simulation3DIntegral as Sim
from ....utils import sdiag, mkvc
import dask
import dask.array as da
from dask.distributed import get_client, Future, Client
from scipy.sparse import csr_matrix as csr


@property
def G(self):
    if getattr(self, "_G", None) is None:
        self._G = self.linear_operator()

    elif isinstance(self._G, Future):
        self._G.result()
        self._G = da.from_zarr(self.sensitivity_path)

    return self._G


@G.setter
def G(self, value):
    assert isinstance(value, (da.Array, Future)) or value is None, f"G must be of one of {array}, {Future} or None. Trying to assign {value}"
    self._G = value


Sim.G = G


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
        diag = ((W[:, None] * self.G) ** 2).sum(axis=0).compute()
        self._gtg_diagonal = diag
    else:
        diag = self._gtg_diagonal
    return mkvc((sdiag(np.sqrt(diag)) @ self.rhoDeriv).power(2).sum(axis=0))


Sim.getJtJdiag = dask_getJtJdiag


def dask_getJ(self, m, f=None):
    """
        Sensitivity matrix
    """

    prod = dask.delayed(csr.dot)(self.G, self.rhoDeriv)

    return da.from_delayed(
        prod, dtype=float, shape=(self.G.shape[0], self.rhoDeriv.shape[1])
    )


Sim.getJ = dask_getJ


def dask_Jvec(self, m, v, f=None):
    """
    Sensitivity times a vector
    """
    dmu_dm_v = self.rhoDeriv @ v
    return da.dot(self.G, dmu_dm_v.astype(np.float32))


Sim.Jvec = dask_Jvec


def dask_Jtvec(self, m, v, f=None):
    """
    Sensitivity transposed times a vector
    """

    Jtvec = da.dot(v, self.G)
    Jtjvec_dmudm = dask.delayed(csr.dot)(Jtvec, self.rhoDeriv)
    h_vec = da.from_delayed(
        Jtjvec_dmudm, dtype=float, shape=[self.rhoDeriv.shape[1]]
    )

    return h_vec


Sim.Jtvec = dask_Jtvec


