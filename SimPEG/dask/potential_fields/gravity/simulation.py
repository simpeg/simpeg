import numpy as np
from ..base import linear_operator
from ....potential_fields.gravity import Simulation3DIntegral as Sim
from ....utils import sdiag, mkvc
from dask import array, delayed
from scipy.sparse import csr_matrix as csr
from dask.distributed import Future, get_client

Sim.linear_operator = linear_operator
Sim.store_sensitivities = "ram"


def dask_fields(self, m):
    """
    Fields computed from a linear operation
    """
    self.model = m
    kernels = self.G # Start process

    if isinstance(kernels, Future):
        client = get_client()
        fields = client.compute(
            client.submit(array.dot, kernels, (self.rhoMap @ m).astype(np.float32))
        )
    else:
        fields = kernels @ (self.rhoMap @ m).astype(np.float32)

    return fields


Sim.fields = dask_fields


def make_row_stack(self):
    n_data_comp = len(self.survey.components)
    components = np.array(list(self.survey.components.keys()))
    active_components = np.hstack(
        [np.c_[values] for values in self.survey.components.values()]
    ).tolist()
    min_hx, min_hy, min_hz = self.mesh.hx.min(), self.mesh.hy.min(), self.mesh.hz.min()
    row = delayed(self.evaluate_integral, pure=True)
    rows = [
        array.from_delayed(
            row(self.Xn, self.Yn, self.Zn, min_hx, min_hy, min_hz, receiver_location, components[component]),
            dtype=np.float32,
            shape=(n_data_comp, self.nC),
        )
        for receiver_location, component in zip(
            self.survey.receiver_locations.tolist(), active_components
        )
    ]

    return array.vstack(rows)


Sim.make_row_stack = make_row_stack


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
        diag = np.einsum('i,ij,ij->j', W, self.G, self.G)

        if isinstance(diag, array.Array):
            diag = diag.compute()

        self._gtg_diagonal = diag

    return mkvc((sdiag(np.sqrt(self._gtg_diagonal)) @ self.rhoDeriv).power(2).sum(axis=0))


Sim.getJtJdiag = dask_getJtJdiag


def dask_Jvec(self, _, v, f=None):
    """
    Sensitivity times a vector
    """

    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    if isinstance(self.Jmatrix, array.Array):
        dmu_dm_v = self.rhoDeriv @ v
        jvec = array.dot(self.Jmatrix, dmu_dm_v.astype(np.float32))
    else:
        dmu_dm_v = self.rhoDeriv @ v
        jvec = self.Jmatrix @ dmu_dm_v.astype(np.float32)

    return jvec


Sim.Jvec = dask_Jvec


def dask_Jtvec(self, _, v, f=None):
    """
    Sensitivity transposed times a vector
    """

    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    if isinstance(self.Jmatrix, array.Array):
        Jtvec = array.dot(v.astype(np.float32), self.Jmatrix)
        Jtjvec_dmudm = delayed(csr.dot)(Jtvec, self.rhoDeriv)
        jt_vec = array.from_delayed(
            Jtjvec_dmudm, dtype=float, shape=[self.rhoDeriv.shape[1]]
        )
    else:
        Jtvec = self.Jmatrix.T @ v.astype(np.float32)
        jt_vec = np.asarray(self.rhoDeriv.T @ Jtvec)

    return jt_vec


Sim.Jtvec = dask_Jtvec


@property
def dask_G(self):
    """
    The linear forward operator
    """
    return self.Jmatrix


Sim.G = dask_G
