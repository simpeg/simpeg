from ...simulation import Jmatrix, dask_Jvec, dask_Jtvec, dask_getJtJdiag
from ..simulation import dpred
from ....electromagnetics.frequency_domain.simulation import BaseFDEMSimulation as Sim
from ....utils import Zero
import numpy as np
import scipy.sparse as sp
import dask.array as da
import zarr

Sim.sensitivity_path = './sensitivity/'
Sim.gtgdiag = None
Sim.store_sensitivities = "ram"
Sim.Jvec = dask_Jvec
Sim.Jtvec = dask_Jtvec
Sim.Jmatrix = Jmatrix
Sim.dpred = dpred
Sim.getJtJdiag = dask_getJtJdiag


def fields(self, m=None, return_Ainv=False):
    if m is not None:
        self.model = m

    f = self.fieldsPair(self)

    Ainv = []
    for freq in self.survey.frequencies:
        A = self.getA(freq)
        rhs = self.getRHS(freq)

        if return_Ainv:
            Ainv += [self.solver(sp.csr_matrix(A.T), **self.solver_opts)]

        Ainv_solve = self.solver(sp.csr_matrix(A), **self.solver_opts)
        u = Ainv_solve * rhs
        Srcs = self.survey.get_sources_by_frequency(freq)
        f[Srcs, self._solutionType] = u

        Ainv_solve.clean()

    if return_Ainv:
        return f, Ainv
    else:
        return f, None


Sim.fields = fields


def compute_J(self, f=None, Ainv=None):

    if f is None:
        f, Ainv = self.fields(self.model, return_Ainv=True)

    self.Ainv = Ainv
    m_size = self.model.size
    row_chunks = int(np.ceil(
        float(self.survey.nD) / np.ceil(float(m_size) * self.survey.nD * 8. * 1e-6 / self.max_chunk_size)
    ))
    if self.store_sensitivities == "disk":
        Jmatrix = zarr.open(
            self.sensitivity_path + f"J.zarr",
            mode='w',
            shape=(self.survey.nD, m_size),
            chunks=(row_chunks, m_size)
        )
    else:
        Jmatrix = np.zeros((self.survey.nD, m_size), dtype=np.float32)

    def eval_store_block(A, freq, df_duT, df_dmT, u_src, src, row_count):
        """
        Evaluate the sensitivities for the block or data and store to zarr
        """
        df_duT = np.hstack(df_duT)
        ATinvdf_duT = (A * df_duT)
        dA_dmT = self.getADeriv(freq, u_src, ATinvdf_duT, adjoint=True)
        dRHS_dmT = self.getRHSDeriv(freq, src, ATinvdf_duT, adjoint=True)
        du_dmT = -dA_dmT
        if not isinstance(dRHS_dmT, Zero):
            du_dmT += dRHS_dmT
        if not isinstance(df_dmT[0], Zero):
            du_dmT += np.hstack(df_dmT)

        block = np.array(du_dmT, dtype=complex).real.T

        if self.store_sensitivities == "disk":
            Jmatrix.set_orthogonal_selection(
                (np.arange(row_count, row_count + block.shape[0]), slice(None)),
                block.astype(np.float32)
            )
        else:
            Jmatrix[row_count: row_count + block.shape[0], :] = (
                block.astype(np.float32)
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
                n_blocs = np.ceil(2 * rx.nD / row_chunks)

                for block in np.array_split(v, n_blocs, axis=1):

                    dfduT, dfdmT = rx.evalDeriv(
                        src, self.mesh, f, v=block, adjoint=True
                    )
                    df_duT += [dfduT]
                    df_dmT += [dfdmT]

                    block_count += dfduT.shape[1]

                    if block_count >= row_chunks:
                        count = eval_store_block(A_i, freq, df_duT, df_dmT, u_src, src, count)
                        df_duT, df_dmT = [], []
                        block_count = 0
                        # blocks, count = store_block(blocks, count)

            if df_duT:
                count = eval_store_block(A_i, freq, df_duT, df_dmT, u_src, src, count)
                block_count = 0

    if len(blocks) != 0:
        if self.store_sensitivities == "disk":
            Jmatrix.set_orthogonal_selection(
                (np.arange(count, self.survey.nD), slice(None)),
                blocks.astype(np.float32)
            )
        else:
            Jmatrix[count: self.survey.nD, :] = (
                blocks.astype(np.float32)
            )

    for A in Ainv:
        A.clean()

    if self.store_sensitivities == "disk":
        del Jmatrix
        return da.from_zarr(self.sensitivity_path + f"J.zarr")
    else:
        return Jmatrix


Sim.compute_J = compute_J
