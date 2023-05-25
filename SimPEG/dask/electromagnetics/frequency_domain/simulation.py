from ....electromagnetics.frequency_domain.simulation import BaseFDEMSimulation as Sim
from ....utils import Zero
import numpy as np
import scipy.sparse as sp

from dask import array, compute, delayed
from SimPEG.dask.simulation import dask_Jvec, dask_Jtvec, dask_getJtJdiag
import zarr

Sim.sensitivity_path = './sensitivity/'
Sim.gtgdiag = None
Sim.store_sensitivities = True

Sim.getJtJdiag = dask_getJtJdiag
Sim.Jvec = dask_Jvec
Sim.Jtvec = dask_Jtvec


def fields(self, m=None, return_Ainv=False):
    if m is not None:
        self.model = m

    f = self.fieldsPair(self)

    Ainv = []
    for freq in self.survey.frequencies:
        A = self.getA(freq)
        rhs = self.getRHS(freq)
        Ainv_solve = self.solver(sp.csr_matrix(A), **self.solver_opts)
        u = Ainv_solve * rhs
        Srcs = self.survey.get_sources_by_frequency(freq)
        f[Srcs, self._solutionType] = u

        Ainv_solve.clean()

        if return_Ainv:
            Ainv += [self.solver(sp.csr_matrix(A.T), **self.solver_opts)]


    if return_Ainv:
        return f, Ainv
    else:
        return f, None


Sim.fields = fields


def compute_J(self, f=None, Ainv=None):

    if f is None:
        f, Ainv = self.fields(self.model, return_Ainv=True)

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

    count = 0
    block_count = 0

    for A_i, freq in zip(Ainv, self.survey.frequencies):

        for ss, src in enumerate(self.survey.get_sources_by_frequency(freq)):
            df_duT, df_dmT = [], []
            blocks_dfduT = []
            blocks_dfdmT = []
            u_src = f[src, self._solutionType]

            col_chunks = int(np.ceil(
                float(self.survey.nD) / np.ceil(float(u_src.shape[0]) * self.survey.nD * 8. * 1e-6 / self.max_chunk_size)
            ))

            for rx in src.receiver_list:
                v = np.eye(rx.nD, dtype=float)
                n_blocs = np.ceil(2 * rx.nD / col_chunks * self.n_cpu)

                for block in np.array_split(v, n_blocs, axis=1):

                    block_count += block.shape[1] * 2
                    blocks_dfduT.append(
                        array.from_delayed(
                            delayed(dfduT, pure=True)(src, rx, self.mesh, f, block),
                            dtype=np.float32,
                            shape=(u_src.shape[0], block.shape[1]*2)
                        )
                    )
                    blocks_dfdmT.append(
                            delayed(dfdmT, pure=True)(src, rx, self.mesh, f, block),
                    )

                    if block_count >= (col_chunks * self.n_cpu):

                        count = parallel_block_compute(self, A_i, Jmatrix, freq, u_src, src, blocks_dfduT, blocks_dfdmT, count, self.n_cpu, m_size)
                        blocks_dfduT = []
                        blocks_dfdmT = []
                        block_count = 0

            if blocks_dfduT:
                count = parallel_block_compute(
                    self, A_i, Jmatrix, freq, u_src, src, blocks_dfduT, blocks_dfdmT, count, self.n_cpu, m_size)
                block_count = 0

    for A in Ainv:
        A.clean()

    if self.store_sensitivities == "disk":
        del Jmatrix
        return array.from_zarr(self.sensitivity_path + f"J.zarr")
    else:
        return Jmatrix


Sim.compute_J = compute_J


def dfduT(source, receiver, mesh, fields, block):
    dfduT, _ = receiver.evalDeriv(
        source, mesh, fields, v=block, adjoint=True
    )

    return dfduT


def dfdmT(source, receiver, mesh, fields, block):
    _, dfdmT = receiver.evalDeriv(
        source, mesh, fields, v=block, adjoint=True
    )

    return dfdmT


def eval_block(simulation, Ainv_deriv_u, frequency, deriv_m, fields, source):
    """
    Evaluate the sensitivities for the block or data and store to zarr
    """
    dA_dmT = simulation.getADeriv(frequency, fields, Ainv_deriv_u, adjoint=True)
    dRHS_dmT = simulation.getRHSDeriv(frequency, source, Ainv_deriv_u, adjoint=True)
    du_dmT = -dA_dmT
    if not isinstance(dRHS_dmT, Zero):
        du_dmT += dRHS_dmT
    if not isinstance(deriv_m, Zero):
        du_dmT += deriv_m

    return np.array(du_dmT, dtype=complex).real.T


def parallel_block_compute(simulation, A_i, Jmatrix, freq, u_src, src, blocks_deriv_u, blocks_deriv_m, counter, sub_threads, m_size):

    field_derivs = array.hstack(blocks_deriv_u).compute()

    # Direct-solver call

    ATinvdf_duT = A_i * field_derivs

    # Even split

    split = np.linspace(0, (ATinvdf_duT.shape[1]) / 2, sub_threads)[1:-1].astype(int) * 2
    sub_blocks_deriv_u = np.array_split(ATinvdf_duT, split, axis=1)

    if isinstance(compute(blocks_deriv_m[0])[0], Zero):
        sub_blocks_dfdmt = [Zero()] * len(sub_blocks_deriv_u)
    else:
        compute_blocks_deriv_m = array.hstack([
            array.from_delayed(
                dfdmT_block,
                dtype=np.float32,
                shape=(u_src.shape[0], dfdmT_block.shape[1] * 2))
            for dfdmT_block in blocks_deriv_m
        ]).compute()
        sub_blocks_dfdmt = np.array_split(compute_blocks_deriv_m, split, axis=1)

    sub_process = []

    for sub_block_dfduT, sub_block_dfdmT in zip(sub_blocks_deriv_u, sub_blocks_dfdmt):
        row_size = int(sub_block_dfduT.shape[1] / 2)
        sub_process.append(
            array.from_delayed(
                delayed(eval_block, pure=True)(
                    simulation,
                    sub_block_dfduT,
                    freq,
                    sub_block_dfdmT,
                    u_src,
                    src
                ),
                dtype=np.float32,
                shape=(row_size, m_size)
            )
        )

    block = array.vstack(sub_process).compute()

    if simulation.store_sensitivities == "disk":
        Jmatrix.set_orthogonal_selection(
            (np.arange(counter, counter + block.shape[0]), slice(None)),
            block.astype(np.float32)
        )
    else:
        Jmatrix[counter: counter + block.shape[0], :] = (
            block.astype(np.float32)
        )

    counter += block.shape[0]
    return counter