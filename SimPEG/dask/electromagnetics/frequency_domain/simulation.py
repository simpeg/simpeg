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
            Ainv += [self.solver(A.T, **self.solver_opts)]

        Ainv_solve = self.solver(A, **self.solver_opts)
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

    def eval_store_block(A, freq, df_duT, df_dmT, u_src, src, row_count):
        """
        Evaluate the sensitivities for the block or data and store to zarr
        """
        df_duT = np.hstack(df_duT)
        ATinvdf_duT = (A * df_duT).reshape((dfduT.shape[0], -1))
        dA_dmT = self.getADeriv(freq, u_src, ATinvdf_duT, adjoint=True)
        dRHS_dmT = self.getRHSDeriv(freq, src, ATinvdf_duT, adjoint=True)
        du_dmT = -dA_dmT
        if not isinstance(dRHS_dmT, Zero):
            du_dmT += dRHS_dmT
        if not isinstance(df_dmT[0], Zero):
            du_dmT += np.hstack(df_dmT)

        block = np.array(du_dmT, dtype=complex).real.T
        Jmatrix.set_orthogonal_selection(
            (np.arange(row_count, row_count + block.shape[0]), slice(None)),
            block
        )
        row_count += block.shape[0]
        return row_count

    blocks = []
    count = 0
    block_count = 0
    for A_i, freq in zip(Ainv, self.survey.frequencies):

        for src in self.survey.get_sources_by_frequency(freq):
            df_duT, df_dmT = [], []
            u_src = f[src, self._solutionType]

            for rx in src.receiver_list:
                v = np.eye(rx.nD, dtype=float)
                n_blocs = np.ceil(rx.nD / row_chunks)

                for block in np.array_split(v, n_blocs, axis=1):

                    dfduT, dfdmT = rx.evalDeriv(
                        src, self.mesh, f, v=block, adjoint=True
                    )
                    df_duT += [dfduT]
                    df_dmT += [dfdmT]

                    block_count += block.shape[1]

                    if block_count >= row_chunks:
                        count = eval_store_block(A_i, freq, df_duT, df_dmT, u_src, src, count)
                        df_duT, df_dmT = [], []
                        block_count = 0
                        # blocks, count = store_block(blocks, count)

            if df_duT:
                count = eval_store_block(A_i, freq, df_duT, df_dmT, u_src, src, count)
                block_count = 0

            # df_duT = np.hstack(df_duT)
            #
            # if df_dmT:
            #     df_dmT = np.hstack(df_dmT)
            # else:
            #     df_dmT = dfdmT
            #
            #
            # ATinvdf_duT = (A_i * df_duT).reshape((dfduT.shape[0], -1))
            # block = []
            #
            # dA_dmT = self.getADeriv(freq, u_src, ATinvdf_duT, adjoint=True)
            # dRHS_dmT = self.getRHSDeriv(freq, src, ATinvdf_duT, adjoint=True)
            # du_dmT = -dA_dmT
            # if not isinstance(dRHS_dmT, Zero):
            #     du_dmT += dRHS_dmT
            # if not isinstance(df_dmT, Zero):
            #     du_dmT += df_dmT
            #
            # block += [np.array(du_dmT, dtype=complex).real.T]
            #
            # block = np.vstack(block)
            # if len(blocks) == 0:
            #     blocks = block.reshape((-1, m_size))
            # else:
            #     blocks = np.vstack([blocks, block])
            # del df_duT, ATinvdf_duT, dA_dmT, dRHS_dmT, du_dmT
                # while blocks.shape[0] >= row_chunks:
                #     Jmatrix.set_orthogonal_selection(
                #         (np.arange(count, count + row_chunks), slice(None)),
                #         blocks[:row_chunks, :]
                #     )
                #     blocks = blocks[row_chunks:, :]
                #     count += row_chunks

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
