from .....electromagnetics.static.resistivity.simulation import BaseDCSimulation as Sim
from .....utils import Zero, mkvc
from .....data import Data
import dask
import dask.array as da
from dask.distributed import Future


Sim.sensitivity_path = './sensitivity/'


def dask_fields(self, m=None, return_Ainv=False):
    if m is not None:
        self.model = m

    f = self.fieldsPair(self)
    A = self.getA()
    Ainv = self.solver(A, **self.solver_opts)
    RHS = self.getRHS()

    f[:, self._solutionType] = Ainv * RHS

    if return_Ainv:
        return (f, Ainv)
    else:
        return (f,)


Sim.fields = dask_fields


def dask_getJtJdiag(self, m, W=None):
    """
        Return the diagonal of JtJ
    """
    self.model = m
    if self.gtgdiag is None:
        if isinstance(self.Jmatrix, Future):
            self.Jmatrix  # Wait to finish
        # Need to check if multiplying weights makes sense
        if W is None:
            self.gtgdiag = da.sum(self.Jmatrix ** 2, axis=0).compute()
        else:
            w = da.from_array(W.diagonal())[:, None]
            self.gtgdiag = da.sum((w * self.Jmatrix) ** 2, axis=0).compute()

    return self.gtgdiag


Sim.getJtJdiag = dask_getJtJdiag


def dask_Jvec(self, m, v):
    """
        Compute sensitivity matrix (J) and vector (v) product.
    """
    self.model = m
    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    return da.dot(self.Jmatrix, v)


Sim.Jvec = dask_Jvec


def dask_Jtvec(self, m, v):
    """
        Compute adjoint sensitivity matrix (J^T) and vector (v) product.
    """
    self.model = m
    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    return da.dot(v, self.Jmatrix)


Sim.Jtvec = dask_Jtvec


def compute_J(self, f=None, Ainv=None):

    if f is None:
        f, Ainv = self.fields(self.model, return_Ainv=True)

    m_size = self.model.size

    blocks = []
    for source in self.survey.source_list:
        u_source = f[source, self._solutionType]
        for rx in source.receiver_list:
            PTv = rx.getP(self.mesh, rx.projGLoc(f)).toarray().T
            df_duTFun = getattr(f, "_{0!s}Deriv".format(rx.projField), None)
            df_duT, df_dmT = df_duTFun(source, None, PTv, adjoint=True)
            ATinvdf_duT = Ainv * df_duT
            dA_dmT = self.getADeriv(u_source, ATinvdf_duT, adjoint=True)
            dRHS_dmT = self.getRHSDeriv(source, ATinvdf_duT, adjoint=True)
            du_dmT = da.from_delayed(
                dask.delayed(-dA_dmT), shape=(m_size, df_duT.shape[1]), dtype=float
            )
            if not isinstance(dRHS_dmT, Zero):
                du_dmT += da.from_delayed(
                    dask.delayed(dRHS_dmT), shape=(m_size, rx.nD), dtype=float
                )
            if not isinstance(df_dmT, Zero):
                du_dmT += da.from_delayed(
                    df_dmT, shape=(m_size, rx.nD), dtype=float
                )

            blocks += [du_dmT.T]

    Jmatrix = da.to_zarr(
        da.vstack(blocks).rechunk('auto'), self.sensitivity_path + "J.zarr",
        compute=True, return_stored=True, overwrite=True
    )

    return Jmatrix


Sim.compute_J = compute_J
