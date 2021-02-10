from ....electromagnetics.frequency_domain.simulation import BaseFDEMSimulation as Sim
from ....utils import Zero, mkvc
import numpy as np
import dask.array as da
from dask.distributed import Future
import zarr
from time import time

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
            Ainv += [self.Solver(A.T, **self.solver_opts)]

        Ainv_solve = self.Solver(A, **self.solver_opts)
        u = Ainv_solve * rhs
        Srcs = self.survey.get_sources_by_frequency(freq)
        f[Srcs, self._solutionType] = u

        Ainv_solve.clean()

    if return_Ainv:
        return f, Ainv
    else:
        return f, None


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
    row_chunks = int(np.ceil(
        float(self.survey.nD) / np.ceil(float(m_size) * self.survey.nD * 8. * 1e-6 / self.max_chunk_size)
    ))
    Jmatrix = zarr.open(
        self.sensitivity_path + f"J.zarr",
        mode='w',
        shape=(self.survey.nD, m_size),
        chunks=(row_chunks, m_size)
    )
    blocks = []
    count = 0
    for A_i, freq in zip(Ainv, self.survey.frequencies):

        for src in self.survey.get_sources_by_frequency(freq):
            u_src = f[src, self._solutionType]

            for rx in src.receiver_list:
                # PTv = rx.getP(self.mesh, rx.projGLoc(f)).toarray().T
                # df_duTFun = getattr(f, "_{0!s}Deriv".format(rx.projField), None)
                # df_duT, df_dmT = df_duTFun(src, None, PTv, adjoint=True)
                # tc = time()
                df_duT, df_dmT = [], []

                for i_datum in range(rx.nD):
                    v = np.zeros(rx.nD, dtype=float)
                    v[i_datum] = 1
                    # PTv = rx.getP(self.mesh, rx.projGLoc(f)).toarray().T
                    # df_duTFun = getattr(f, "_{0!s}Deriv".format(rx.projField), None)
                    # df_duT, df_dmT = df_duTFun(src, None, PTv, adjoint=True)
                    dfduT, dfdmT = rx.evalDeriv(
                        src, self.mesh, f, v=v, adjoint=True
                    )
                    df_duT += [dfduT]

                    if not isinstance(dfdmT, Zero):
                        df_dmT += [dfdmT]

                # print(f"First loop {time()-tc}")

                df_duT = np.hstack(df_duT)

                if df_dmT:
                    df_dmT = np.hstack(df_dmT)
                else:
                    df_dmT = dfdmT

                # tc = time()
                ATinvdf_duT = (A_i * df_duT).reshape((dfduT.shape[0], -1))
                # print(f"Solve {time() - tc}")
                nc = int(ATinvdf_duT.shape[1]/rx.nD)
                # tc = time()
                block = []
                for ii in range(rx.nD):
                    dA_dmT = self.getADeriv(freq, u_src, ATinvdf_duT[:, ii*nc:ii*nc+nc], adjoint=True)
                    dRHS_dmT = self.getRHSDeriv(freq, src, ATinvdf_duT[:, ii*nc:ii*nc+nc], adjoint=True)
                    du_dmT = -dA_dmT
                    if not isinstance(dRHS_dmT, Zero):
                        du_dmT += dRHS_dmT
                    if not isinstance(df_dmT, Zero):
                        du_dmT += df_dmT

                    # if rx.component == "real":
                    #     block = np.array(du_dmT, dtype=complex).real.T.reshape((-1, m_size))
                    # elif rx.component == "imag":
                    #     block = -np.array(du_dmT, dtype=complex).real.T.reshape((-1, m_size))

                    block += [np.array(du_dmT, dtype=complex).real.T]
                # print(f"Second looop {time() - tc}")
                block = np.vstack(block)
                if len(blocks) == 0:
                    blocks = block.reshape((-1, m_size))
                else:
                    blocks = np.vstack([blocks, block])
                del df_duT, ATinvdf_duT, dA_dmT, dRHS_dmT, du_dmT
                while blocks.shape[0] >= row_chunks:
                    Jmatrix.set_orthogonal_selection(
                        (np.arange(count, count + row_chunks), slice(None)),
                        blocks[:row_chunks, :]
                    )
                    blocks = blocks[row_chunks:, :]
                    count += row_chunks

    if len(blocks) != 0:
        Jmatrix.set_orthogonal_selection(
            (np.arange(count, self.survey.nD), slice(None)),
            blocks
        )

    del Jmatrix
    for A in Ainv:
        A.clean()

    return da.from_zarr(self.sensitivity_path + f"J.zarr")


Sim.compute_J = compute_J
