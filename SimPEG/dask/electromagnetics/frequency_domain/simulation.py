from ....electromagnetics.frequency_domain.simulation import BaseFDEMSimulation as Sim
from ....utils import Zero, mkvc
import numpy as np
from ....data import Data
import dask
import dask.array as da
from dask.distributed import Future


Sim.sensitivity_path = './sensitivity/'
Sim.gtgdiag = None

def fields(self, m=None, return_Ainv=False):
    if m is not None:
        self.model = m

    f = self.fieldsPair(self)

    Ainv = []
    for freq in self.survey.frequencies:
        A = self.getA(freq)
        rhs = self.getRHS(freq)

        if return_Ainv:
            Ainv += [self.Solver(A, **self.solver_opts)]
            u = Ainv[-1] * rhs
        else:
            Ainv = self.Solver(A, **self.solver_opts)
            u = Ainv * rhs
        Srcs = self.survey.get_sources_by_frequency(freq)
        f[Srcs, self._solutionType] = u

    if return_Ainv:
        return (f, Ainv)
    else:
        return (f,)


Sim.fields = fields


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
    count = -1
    for A_i, freq in zip(Ainv, self.survey.frequencies):

        for src in self.survey.get_sources_by_frequency(freq):
            u_src = f[src, self._solutionType]

            for rx in src.receiver_list:
                # PTv = rx.getP(self.mesh, rx.projGLoc(f)).toarray().T
                # df_duTFun = getattr(f, "_{0!s}Deriv".format(rx.projField), None)
                # df_duT, df_dmT = df_duTFun(src, None, PTv, adjoint=True)
                count += 1

                df_duT, df_dmT = rx.evalDeriv(
                    src, self.mesh, f, v=np.ones(1), adjoint=True
                )

                ATinvdf_duT = A_i * df_duT
                dA_dmT = self.getADeriv(freq, u_src, ATinvdf_duT, adjoint=True)
                dRHS_dmT = self.getRHSDeriv(freq, src, ATinvdf_duT, adjoint=True)
                du_dmT = -dA_dmT
                if not isinstance(dRHS_dmT, Zero):
                    du_dmT += dRHS_dmT
                if not isinstance(df_dmT, Zero):
                    du_dmT += df_dmT

                if rx.component == "real":
                    blocks += [np.array(du_dmT, dtype=complex).real.T]
                elif rx.component == "imag":
                    blocks += [-np.array(du_dmT, dtype=complex).real.T]


    Jmatrix = da.to_zarr(
        da.vstack(blocks).rechunk('auto'), self.sensitivity_path + "J.zarr",
        compute=True, return_stored=True, overwrite=True
    )

    return Jmatrix


Sim.compute_J = compute_J
