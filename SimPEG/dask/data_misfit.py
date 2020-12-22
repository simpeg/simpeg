from ..data_misfit import L2DataMisfit
import dask.array as da


def dask_call(self, m, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.__call__`
    """
    R = self.W * self.residual(m, f=f)
    phi_d = 0.5 * da.dot(R, R)
    return self.client.compute(phi_d, workers=self.workers)


L2DataMisfit.__call__ = dask_call


def dask_deriv(self, m, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.deriv`
    """

    if f is None:
        if self.simulation._Jmatrix is None:
            f = self.simulation.fields(m)

    wtw_d = self.W.diagonal() ** 2.0 * self.residual(m, f=f)

    return self.client.compute(self.simulation.Jtvec(m, wtw_d, f=f), workers=self.workers)


L2DataMisfit.deriv = dask_deriv


def dask_deriv2(self, m, v, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.deriv2`
    """

    if f is None:
        if self.simulation._Jmatrix is None:
            f = self.simulation.fields(m)

    jvec = self.simulation.Jvec(m, v, f=f)
    w_jvec = self.W.diagonal() ** 2.0 * jvec

    return self.client.compute(
            self.simulation.Jtvec(m, w_jvec, f=f), workers=self.workers
        )


L2DataMisfit.deriv2 = dask_deriv2
