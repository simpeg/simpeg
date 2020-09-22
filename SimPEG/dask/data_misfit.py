from ..data_misfit import L2DataMisfit

import dask
import dask.array as da
from scipy.sparse import csr_matrix as csr


def dask_deriv(self, m, f=None):
    """
    deriv(m, f=None)
    Derivative of the data misfit

    .. math::

        \mathbf{J}^{\top} \mathbf{W}^{\top} \mathbf{W}
        (\mathbf{d} - \mathbf{d}^{obs})

    :param numpy.ndarray m: model
    :param SimPEG.fields.Fields f: fields object
    """

    if f is None:
        f = self.simulation.fields(m)

    w_d = dask.delayed(csr.dot)(self.W, self.residual(m, f=f))

    wtw_d = dask.delayed(csr.dot)(self.scale * w_d, self.W)

    row = da.from_delayed(wtw_d, dtype=float, shape=[self.W.shape[0]])
    return self.prob.Jtvec(m, row, f=f)


L2DataMisfit.deriv = dask_deriv


def dask_deriv2(self, m, v, f=None):
    """
    deriv2(m, v, f=None)

    .. math::

        \mathbf{J}^{\top} \mathbf{W}^{\top} \mathbf{W} \mathbf{J}

    :param numpy.ndarray m: model
    :param numpy.ndarray v: vector
    :param SimPEG.fields.Fields f: fields object
    """

    if f is None:
        f = self.simulation.fields(m)

    jtvec = self.simulation.Jtvec_approx(m, v, f=f)

    w_jtvec = dask.delayed(csr.dot)(self.W, jtvec)

    wtw_jtvec = dask.delayed(csr.dot)(w_jtvec, self.W)

    row = da.from_delayed(wtw_jtvec, dtype=float, shape=[self.W.shape[0]])
    return self.simulation.Jtvec_approx(m, row, f=f)


L2DataMisfit.deriv2 = dask_deriv2
