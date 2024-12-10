from ..simulation import BaseSimulation as Sim

from dask import array, delayed

import numpy as np


Sim._max_ram = 16


@property
def max_ram(self):
    "Maximum ram in (Gb)"
    return self._max_ram


@max_ram.setter
def max_ram(self, other):
    if other <= 0:
        raise ValueError("max_ram must be greater than 0")
    self._max_ram = other


Sim.max_ram = max_ram

Sim._max_chunk_size = 128


@property
def max_chunk_size(self):
    "Largest chunk size (Mb) used by Dask"
    return self._max_chunk_size


@max_chunk_size.setter
def max_chunk_size(self, other):
    if other <= 0:
        raise ValueError("max_chunk_size must be greater than 0")
    self._max_chunk_size = other


Sim.max_chunk_size = max_chunk_size


def dask_getJtJdiag(self, m, W=None, f=None):
    """
    Return the diagonal of JtJ
    """
    if getattr(self, "_jtjdiag", None) is None:
        self.model = m
        if W is None:
            W = np.ones(self.Jmatrix.shape[0])
        else:
            W = W.diagonal()

        self._jtj_diag = array.einsum("i,ij,ij->j", W**2, self.Jmatrix, self.Jmatrix)

    return self._jtj_diag


Sim.getJtJdiag = dask_getJtJdiag


def dask_dpred(self, m=None, f=None):
    r"""
    dpred(m, f=None)
    Create the projected data from a model.
    The fields, f, (if provided) will be used for the predicted data
    instead of recalculating the fields (which may be expensive!).

    .. math::

        d_\\text{pred} = P(f(m))

    Where P is a projection of the fields onto the data space.
    """
    if self.survey is None:
        raise AttributeError(
            "The survey has not yet been set and is required to compute "
            "data. Please set the survey for the simulation: "
            "simulation.survey = survey"
        )

    if f is None:
        if m is None:
            m = self.model
        f = self.fields(m)

    def evaluate_receiver(source, receiver, mesh, fields):
        return receiver.eval(source, mesh, fields).flatten()

    row = delayed(evaluate_receiver, pure=True)
    rows = []
    for src in self.survey.source_list:
        for rx in src.receiver_list:
            rows.append(
                array.from_delayed(
                    row(src, rx, self.mesh, f),
                    dtype=np.float32,
                    shape=(rx.nD,),
                )
            )

    data = array.hstack(rows).compute()

    return data


Sim.dpred = dask_dpred


def dask_Jvec(self, m, v, **_):
    """
    Compute sensitivity matrix (J) and vector (v) product.
    """
    self.model = m

    if isinstance(self.Jmatrix, np.ndarray):
        return self.Jmatrix @ v.astype(np.float32)

    return array.dot(self.Jmatrix, v).astype(np.float32)


def dask_Jtvec(self, m, v, **_):
    """
    Compute adjoint sensitivity matrix (J^T) and vector (v) product.
    """
    self.model = m

    if isinstance(self.Jmatrix, np.ndarray):
        return self.Jmatrix.T @ v.astype(np.float32)

    return array.dot(v, self.Jmatrix).astype(np.float32)


@property
def Jmatrix(self):
    """
    Sensitivity matrix stored on disk
    Return the diagonal of JtJ
    """
    if getattr(self, "_Jmatrix", None) is None:
        self._Jmatrix = self.compute_J(self.model)

    return self._Jmatrix


Sim.Jmatrix = Jmatrix
