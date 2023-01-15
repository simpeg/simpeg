import numpy as np

from ..data_misfit import L2DataMisfit
from ..fields import Fields
from ..utils import mkvc
from .utils import compute
import dask.array as da
from scipy.sparse import csr_matrix as csr
from dask import delayed


def dask_call(self, m, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.__call__`
    """
    R = self.W * self.residual(m, f=f)
    phi_d = 0.5 * da.dot(R, R)
    if not isinstance(phi_d, np.ndarray):
        return compute(self, phi_d)
    return phi_d

L2DataMisfit.__call__ = dask_call


def dask_deriv(self, m, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.deriv`
    """

    if getattr(self, "model_map", None) is not None:
        m = self.model_map @ m

    wtw_d = self.W.diagonal() ** 2.0 * self.residual(m, f=f)
    Jtvec = compute(self, self.simulation.Jtvec(m, wtw_d))

    if getattr(self, "model_map", None) is not None:
        Jtjvec_dmudm = delayed(csr.dot)(Jtvec, self.model_map.deriv(m))
        h_vec = da.from_delayed(
            Jtjvec_dmudm, dtype=float, shape=[self.model_map.deriv(m).shape[1]]
        )
        if not isinstance(h_vec, np.ndarray):
            return compute(self, h_vec)
        return h_vec

    if not isinstance(Jtvec, np.ndarray):
        return compute(self, Jtvec)
    return Jtvec


L2DataMisfit.deriv = dask_deriv


def dask_deriv2(self, m, v, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.deriv2`
    """

    if getattr(self, "model_map", None) is not None:
        m = self.model_map @ m
        v = self.model_map.deriv(m) @ v

    jvec = compute(self, self.simulation.Jvec(m, v))
    w_jvec = self.W.diagonal() ** 2.0 * jvec
    jtwjvec = compute(self, self.simulation.Jtvec(m, w_jvec))

    if getattr(self, "model_map", None) is not None:
        Jtjvec_dmudm = delayed(csr.dot)(jtwjvec, self.model_map.deriv(m))
        h_vec = da.from_delayed(
            Jtjvec_dmudm, dtype=float, shape=[self.model_map.deriv(m).shape[1]]
        )
        if not isinstance(h_vec, np.ndarray):
            return compute(self, h_vec)
        return h_vec

    if not isinstance(jtwjvec, np.ndarray):
        return compute(self, jtwjvec)
    return jtwjvec


L2DataMisfit.deriv2 = dask_deriv2


def dask_residual(self, m, f=None):
    if self.data is None:
        raise Exception("data must be set before a residual can be calculated.")

    if isinstance(f, Fields) or f is None:
        return self.simulation.residual(m, self.data.dobs, f=f)
    elif f.shape == self.data.dobs.shape:
        return mkvc(f - self.data.dobs)
    else:
        raise Exception(f"Attribute f must be or type {Fields}, numpy.array or None.")


L2DataMisfit.residual = dask_residual