from .....electromagnetics.static.induced_polarization.simulation import (
    BaseIPSimulation as Sim,
)

from ..resistivity.simulation import (
    compute_J,
    getSourceTerm,
)


from .....data import Data
import dask.array as da
from dask.distributed import Future
import numpy as np
import numcodecs

numcodecs.blosc.use_threads = False


def fields(self, m=None, return_Ainv=False):
    if m is not None:
        self.model = m

    A = self.getA()
    Ainv = self.solver(A, **self.solver_opts)
    RHS = self.getRHS()

    f = self.fieldsPair(self)
    f[:, self._solutionType] = Ainv * np.asarray(RHS.todense())

    if self._scale is None:
        scale = Data(self.survey, np.ones(self.survey.nD))
        # loop through receivers to check if they need to set the _dc_voltage
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                if (
                    rx.data_type == "apparent_chargeability"
                    or self._data_type == "apparent_chargeability"
                ):
                    scale[src, rx] = 1.0 / rx.eval(src, self.mesh, f)
        self._scale = scale.dobs

    self._stashed_fields = f
    if return_Ainv:
        return f, Ainv
    return f


def dpred(self, m=None, f=None):
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

    data = self.Jvec(m, m)

    return np.asarray(data)


def getJtJdiag(self, m, W=None, f=None):
    """
    Return the diagonal of JtJ
    """
    self.model = m
    if getattr(self, "_jtjdiag", None) is None:

        if W is None:
            W = self._scale * np.ones(self.nD)
        else:
            W = (self._scale * W.diagonal()) ** 2.0

        diag = np.einsum("i,ij,ij->j", W, self.Jmatrix, self.Jmatrix)

        self._jtjdiag = diag

    return self._jtjdiag


def Jvec(self, m, v, f=None):
    """
    Compute sensitivity matrix (J) and vector (v) product.
    """
    self.model = m

    if isinstance(self.Jmatrix, np.ndarray):
        return self._scale.astype(np.float32) * (self.Jmatrix @ v.astype(np.float32))

    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    return self._scale.astype(np.float32) * da.dot(self.Jmatrix, v).astype(np.float32)


def Jtvec(self, m, v, f=None):
    """
    Compute adjoint sensitivity matrix (J^T) and vector (v) product.
    """
    self.model = m

    if isinstance(self.Jmatrix, np.ndarray):
        return (self._scale * v.astype(np.float32)).astype(np.float32) @ self.Jmatrix

    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    return da.dot((v * self._scale).astype(np.float32), self.Jmatrix).astype(np.float32)


Sim.compute_J = compute_J
Sim.getSourceTerm = getSourceTerm
Sim.Jtvec = Jtvec
Sim.Jvec = Jvec
Sim.getJtJdiag = getJtJdiag
Sim.dpred = dpred
Sim.fields = fields
