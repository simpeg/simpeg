from ....electromagnetics.time_domain.simulation import BaseTDEMSimulation as Sim
from ....utils import Zero
import numpy as np
import scipy.sparse as sp

from dask import array, compute, delayed
from SimPEG.dask.simulation import dask_Jvec, dask_Jtvec, dask_getJtJdiag
import zarr
from SimPEG.utils import mkvc
Sim.sensitivity_path = './sensitivity/'
Sim.gtgdiag = None
Sim.store_sensitivities = True

Sim.getJtJdiag = dask_getJtJdiag
Sim.Jvec = dask_Jvec
Sim.Jtvec = dask_Jtvec
Sim.clean_on_model_update = ["_Jmatrix", "gtgdiag"]


def fields(self, m=None, return_Ainv=False):
    if m is not None:
        self.model = m

    f = self.fieldsPair(self)

    # set initial fields
    f[:, self._fieldType + "Solution", 0] = self.getInitialFields()

    Ainv = {}
    ATinv = {}
    for tInd, dt in enumerate(self.time_steps):

        if dt not in Ainv:
            A = self.getAdiag(tInd)
            Ainv[dt] = self.solver(sp.csr_matrix(A), **self.solver_opts)
            if return_Ainv:
                ATinv[dt] = self.solver(sp.csr_matrix(A.T), **self.solver_opts)

        rhs = self.getRHS(tInd + 1)
        Asubdiag = self.getAsubdiag(tInd)
        sol = Ainv[dt] * (rhs - Asubdiag * f[:, (self._fieldType + "Solution"), tInd])
        f[:, self._fieldType + "Solution", tInd + 1] = sol

    for A in Ainv.values():
        A.clean()

    if return_Ainv:
        return f, ATinv
    else:
        return f, None


Sim.fields = fields


def dask_dpred(self, m=None, f=None, compute_J=False):
    """
    dpred(m, f=None)
    Create the projected data from a model.
    The fields, f, (if provided) will be used for the predicted data
    instead of recalculating the fields (which may be expensive!).

    .. math::

        d_\\text{pred} = P(f(m))

    Where P is a projection of the fields onto the data space.
    """
    if self.survey is None:
        raise AttributeError(
            "The survey has not yet been set and is required to compute "
            "data. Please set the survey for the simulation: "
            "simulation.survey = survey"
        )

    if f is None:
        if m is None:
            m = self.model
        f, Ainv = self.fields(m, return_Ainv=compute_J)

    def evaluate_receiver(source, receiver, mesh, time_mesh, fields):
        return receiver.eval(source, mesh, time_mesh, fields).flatten()

    row = delayed(evaluate_receiver, pure=True)
    rows = []
    for src in self.survey.source_list:
        for rx in src.receiver_list:
            rows.append(array.from_delayed(
                row(src, rx, self.mesh, self.time_mesh, f),
                dtype=np.float32,
                shape=(rx.nD,),
            ))

    data = array.hstack(rows).compute()

    if compute_J and self._Jmatrix is None:
        Jmatrix = self.compute_J(f=f, Ainv=Ainv)
        return data, Jmatrix

    return data


Sim.dpred = dask_dpred


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
    #
    # count = 0
    # block_count = 0
    #
    # for tInd, dt in enumerate(self.time_steps):
    #
    #     for ss, src in enumerate(self.survey.source_list):
    #         df_duT, df_dmT = [], []
    #         blocks_dfduT = []
    #         blocks_dfdmT = []
    #         u_src = f[src, self._fieldType, :]
    #
    #         col_chunks = int(np.ceil(
    #             float(self.survey.nD) / np.ceil(float(u_src.shape[0]) * self.survey.nD * 8. * 1e-6 / self.max_chunk_size)
    #         ))
    #
    #         for rx in src.receiver_list:
    #             v = np.eye(rx.nD, dtype=float)
    #             n_blocs = np.ceil(2 * rx.nD / col_chunks * self.n_cpu)
    #
    #             for block in np.array_split(v, n_blocs, axis=1):
    #
    #                 block_count += block.shape[1] * 2
    #                 blocks_dfduT.append(
    #                     array.from_delayed(
    #                         delayed(dfduT, pure=True)(src, rx, self.mesh, self.time_mesh, f, block),
    #                         dtype=np.float32,
    #                         shape=(u_src.shape[0], block.shape[1]*2)
    #                     )
    #                 )
    #                 blocks_dfdmT.append(
    #                         delayed(dfdmT, pure=True)(src, rx, self.mesh, f, block),
    #                 )
    #
    #                 if block_count >= (col_chunks * self.n_cpu):
    #
    #                     count = parallel_block_compute(self, Ainv[dt], Jmatrix, freq, u_src, src, blocks_dfduT, blocks_dfdmT, count, self.n_cpu, m_size)
    #                     blocks_dfduT = []
    #                     blocks_dfdmT = []
    #                     block_count = 0
    #
    #         if blocks_dfduT:
    #             count = parallel_block_compute(
    #                 self, A_i, Jmatrix, freq, u_src, src, blocks_dfduT, blocks_dfdmT, count, self.n_cpu, m_size)
    #             block_count = 0
    #
    # for A in Ainv:
    #     A.clean()
    #
    # if self.store_sensitivities == "disk":
    #     del Jmatrix
    #     return array.from_zarr(self.sensitivity_path + f"J.zarr")
    # else:
    #     return Jmatrix
    solution_type = self._fieldType + "Solution"  # the thing we solved for
    field_type = "{}Deriv".format(self._fieldType)

    # Ensure v is a data object.
    # if not isinstance(v, Data):
    #     v = Data(self.survey, v)

    df_duT_v = []

    # same size as fields at a single timestep
    ATinv_df_duT_v = np.zeros(
        (
            len(self.survey.source_list),
            len(f[self.survey.source_list[0], solution_type, 0]),
        ),
        dtype=float,
    )

    # Loop over sources and receivers to create a fields object:
    # PT_v, df_duT_v, df_dmT_v
    # initialize storage for PT_v (don't need to preserve over sources)
    # PT_v = self.Fields_Derivs(self)
    block_size = len(f[self.survey.source_list[0], solution_type, 0])
    d_count = 0
    for i_s, src in enumerate(self.survey.source_list):
        for rx in src.receiver_list:
            v = np.eye(rx.nD, dtype=float)
            PT_v = rx.evalDeriv(
                src, self.mesh, self.time_mesh, f, v, adjoint=True
            )
            df_duTFun = getattr(f, "_{}Deriv".format(rx.projField), None)
            cur = df_duTFun(
                self.nT,
                src,
                None,
                PT_v[-block_size:, :],
                adjoint=True,
            )
            df_duT_past = cur[0]

            if not isinstance(cur[1], Zero):
                Jmatrix[d_count:d_count+rx.nD, :] += cur[1].T

            for tInd, dt in zip(reversed(range(self.nT)), reversed(self.time_steps)):
                AdiagTinv = Ainv[dt]

                if tInd >= self.nT - 1:
                    ATinv_df_duT_v = AdiagTinv * df_duT_past
                else:
                    Asubdiag = self.getAsubdiag(tInd+1)
                    ATinv_df_duT_v = AdiagTinv * (
                            df_duT_past
                            - Asubdiag.T * ATinv_df_duT_v
                    )

                dAsubdiagT_dm_v = self.getAsubdiagDeriv(
                    tInd, f[src, solution_type, tInd], ATinv_df_duT_v, adjoint=True
                )

                dRHST_dm_v = self.getRHSDeriv(
                    tInd + 1, src, ATinv_df_duT_v, adjoint=True
                )
                un_src = f[src, solution_type, tInd + 1]
                dAT_dm_v = self.getAdiagDeriv(
                    tInd, un_src, ATinv_df_duT_v, adjoint=True
                )
                Jmatrix[d_count:d_count+rx.nD, :] += (-dAT_dm_v - dAsubdiagT_dm_v + dRHST_dm_v).T
                cur = df_duTFun(
                    tInd,
                    src,
                    None,
                    PT_v[(tInd * block_size):((tInd+1) * block_size), :],
                    adjoint=True,
                )
                df_duT_past = cur[0]

                if not isinstance(cur[1], Zero):
                    Jmatrix[d_count:d_count+rx.nD, :] += cur[1].T

            d_count += rx.nD

    for A in Ainv.values():
        A.clean()

    if self.store_sensitivities == "disk":
        del Jmatrix
        return array.from_zarr(self.sensitivity_path + f"J.zarr")
    else:
        return Jmatrix

Sim.compute_J = compute_J


def dfduT(source, receiver, mesh, time_mesh, fields, block):
    dfduT, _ = receiver.evalDeriv(
        source, mesh, time_mesh, fields, v=block, adjoint=True
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