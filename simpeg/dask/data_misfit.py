import numpy as np

from ..data_misfit import L2DataMisfit

from ..utils import mkvc

from dask.distributed import get_client, Future


def _data_residual(dpred, dobs):
    return mkvc(dpred) - dobs


def _misfit(residual, W):
    vec = W * residual
    return np.dot(vec, vec)


def dask_call(self, m, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.__call__`
    """
    dpred = self.simulation.dpred(m, f=f)

    if isinstance(dpred, Future):
        client = get_client()
        residuals = client.submit(_data_residual, dpred, self.data.dobs)
        phi_d = client.submit(_misfit, residuals, self.W)
    else:
        residuals = _data_residual(dpred, self.data.dobs)
        phi_d = _misfit(residuals, self.W)

    return phi_d


L2DataMisfit.__call__ = dask_call


def dask_deriv(self, m, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.deriv`
    """
    wtw_d = self.W.diagonal() ** 2.0 * self.residual(m, f=f)
    Jtvec = self, self.simulation.Jtvec(m, wtw_d)

    return Jtvec


L2DataMisfit.deriv = dask_deriv


def _stack_futures(futures, W):
    return W * np.concatenate(futures)


def dask_deriv2(self, m, v, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.deriv2`
    """
    jvec = self.simulation.Jvec(m, v)
    if isinstance(jvec, Future):
        client = get_client()
        w_jvec = client.submit(_stack_futures, jvec, self.W.diagonal() ** 2.0)

    else:
        w_jvec = self.W.diagonal() ** 2.0 * jvec

    jtwjvec = self.simulation.Jtvec(m, w_jvec)

    return jtwjvec


L2DataMisfit.deriv2 = dask_deriv2
