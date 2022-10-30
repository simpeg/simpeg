import numpy as np
from ..base import linear_operator
from ....potential_fields.magnetics import Simulation3DIntegral as Sim
from ....utils import sdiag, mkvc
from scipy.sparse import csr_matrix as csr
from dask.distributed import get_client, Future
from dask import delayed, array


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
            client.submit(array.dot, kernels, (self.chiMap @ m).astype(np.float32))
        )
    else:
        fields = kernels @ (self.chiMap @ m).astype(np.float32)

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
            row(self.Xn, self.Yn, self.Zn, min_hx, min_hy, min_hz, self.M, self.tmi_projection, receiver_location, components[component]),
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

    if W is None:
        W = np.ones(self.nD)
    else:
        W = W.diagonal()
    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish
    if getattr(self, "_gtg_diagonal", None) is None:
        if not self.is_amplitude_data:
            diag = np.einsum('i,ij,ij->j', W, self.G, self.G)
        else:  # self.modelType is amplitude
            fieldDeriv = self.fieldDeriv
            J = (
                fieldDeriv[0, :, None] * self.G[::3]
                + fieldDeriv[1, :, None] * self.G[1::3]
                + fieldDeriv[2, :, None] * self.G[2::3]
            )
            diag = ((W[:, None] * J) ** 2).sum(axis=0)

        if isinstance(diag, array.Array):
            diag = diag.compute()

        self._gtg_diagonal = diag

    return mkvc((sdiag(np.sqrt(self._gtg_diagonal)) @ self.chiDeriv).power(2).sum(axis=0))


Sim.getJtJdiag = dask_getJtJdiag


def dask_Jvec(self, m, v, f=None):
    """
    Sensitivity times a vector
    """
    self.model = m
    dmu_dm_v = self.chiDeriv @ v

    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    if isinstance(self.Jmatrix, array.Array):
        jvec = array.dot(self.Jmatrix, dmu_dm_v.astype(np.float32))

    else:
        jvec = self.Jmatrix @ dmu_dm_v.astype(np.float32)

    if self.is_amplitude_data:
        jvec = jvec.reshape((-1, 3)).T
        fieldDeriv_jvec = self.fieldDeriv * jvec
        return fieldDeriv_jvec[0] + fieldDeriv_jvec[1] + fieldDeriv_jvec[2]

    return jvec


Sim.Jvec = dask_Jvec


def dask_Jtvec(self, m, v, f=None):
    """
    Sensitivity transposed times a vector
    """
    self.model = m

    if self.is_amplitude_data:
        v = (self.fieldDeriv * v).T.reshape(-1)

    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    if isinstance(self.Jmatrix, array.Array):
        Jtvec = array.dot(v.astype(np.float32), self.Jmatrix)
        Jtjvec_dmudm = delayed(csr.dot)(Jtvec, self.chiDeriv)
        jt_vec = array.from_delayed(
            Jtjvec_dmudm, dtype=float, shape=[self.chiDeriv.shape[1]]
        )
    else:
        jt_vec = self.Jmatrix.T @ v.astype(np.float32)

    return jt_vec


Sim.Jtvec = dask_Jtvec


@property
def dask_G(self):
    """
    The linear forward operator
    """
    return self.Jmatrix


Sim.G = dask_G
