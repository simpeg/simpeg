from ..data_misfit import L2DataMisfit
import dask.array as da
from scipy.sparse import csr_matrix as csr
from dask import delayed


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

    if getattr(self, "model_map", None) is not None:
        m = self.model_map @ m

    if getattr(self.simulation, "_Jmatrix", None) is None:
        self.simulation.fields(m)

    wtw_d = self.W.diagonal() ** 2.0 * self.residual(m, f=f)

    Jtvec = self.simulation.Jtvec(m, wtw_d)

    if getattr(self, "model_map", None) is not None:
        Jtjvec_dmudm = delayed(csr.dot)(Jtvec, self.model_map.deriv(m))
        h_vec = da.from_delayed(
            Jtjvec_dmudm, dtype=float, shape=[self.model_map.deriv(m).shape[1]]
        )
        return self.client.compute(h_vec, workers=self.workers)

    return self.client.compute(Jtvec, workers=self.workers)


L2DataMisfit.deriv = dask_deriv


def dask_deriv2(self, m, v, f=None):
    """
    Distributed :obj:`simpeg.data_misfit.L2DataMisfit.deriv2`
    """

    if getattr(self, "model_map", None) is not None:
        m = self.model_map @ m
        v = self.model_map.deriv(m) @ v

    if getattr(self.simulation, "_Jmatrix", None) is None:
        self.simulation.fields(m)

    jvec = self.simulation.Jvec(m, v)
    w_jvec = self.W.diagonal() ** 2.0 * jvec
    jtwjvec = self.simulation.Jtvec(m, w_jvec)

    if getattr(self, "model_map", None) is not None:
        Jtjvec_dmudm = delayed(csr.dot)(jtwjvec, self.model_map.deriv(m))
        h_vec = da.from_delayed(
            Jtjvec_dmudm, dtype=float, shape=[self.model_map.deriv(m).shape[1]]
        )
        return self.client.compute(h_vec, workers=self.workers)

    return self.client.compute(jtwjvec, workers=self.workers)


L2DataMisfit.deriv2 = dask_deriv2
