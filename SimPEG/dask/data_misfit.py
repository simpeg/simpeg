from ..data_misfit import L2DataMisfit
import dask.array as da



def dask_call(self, m, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.__call__`
    """
    R = self.W * self.residual(m, f=f)
    phi_d = da.dot(R, R)
    return self.client.compute(phi_d, workers=self.workers)


L2DataMisfit.__call__ = dask_call


def dask_deriv(self, m, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.deriv`
    """
    if f is None:
        f = self.simulation.fields(m)

    wtw_d = self.W.T * (self.W * self.residual(m, f=f))

    return self.client.compute(self.simulation.Jtvec(m, wtw_d, f=f), workers=self.workers)


L2DataMisfit.deriv = dask_deriv


def dask_deriv2(self, m, v, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.deriv2`
    """

    if f is None:
        f = self.simulation.fields(m)

    jtvec = self.simulation.Jtvec(m, v, f=f, workers=self.workers)
    w_jtvec = self.W ** 2.0 * jtvec

    return self.client.compute(
            self.simulation.Jtvec(m, w_jtvec, f=f), workers=self.workers
        )


L2DataMisfit.deriv2 = dask_deriv2
