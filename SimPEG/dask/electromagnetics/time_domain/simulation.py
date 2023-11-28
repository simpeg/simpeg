import dask
import dask.array

from ....electromagnetics.time_domain.simulation import BaseTDEMSimulation as Sim
from ....utils import Zero
from multiprocessing import cpu_count
import numpy as np
import scipy.sparse as sp
from dask import array, delayed
from SimPEG.dask.simulation import dask_Jvec, dask_Jtvec, dask_getJtJdiag
import zarr
from time import time
from tqdm import tqdm

Sim.sensitivity_path = "./sensitivity/"
Sim.store_sensitivities = "ram"

Sim.getJtJdiag = dask_getJtJdiag
Sim.Jvec = dask_Jvec
Sim.Jtvec = dask_Jtvec
Sim.clean_on_model_update = ["_Jmatrix", "_jtjdiag"]


def fields(self, m=None, return_Ainv=False):
    if m is not None:
        self.model = m

    f = self.fieldsPair(self)
    f[:, self._fieldType + "Solution", 0] = self.getInitialFields()
    Ainv = {}
    ATinv = {}

    for tInd, dt in enumerate(self.time_steps):
        if dt not in Ainv:
            A = self.getAdiag(tInd)
            Ainv[dt] = self.solver(sp.csr_matrix(A), **self.solver_opts)
            if return_Ainv:
                ATinv[dt] = self.solver(sp.csr_matrix(A.T), **self.solver_opts)

        Asubdiag = self.getAsubdiag(tInd)
        rhs = -Asubdiag * f[:, (self._fieldType + "Solution"), tInd]

        if (
            np.abs(self.survey.source_list[0].waveform.eval(self.times[tInd + 1]))
            > 1e-8
        ):
            rhs += self.getRHS(tInd + 1)

        sol = Ainv[dt] * rhs
        f[:, self._fieldType + "Solution", tInd + 1] = sol

    for A in Ainv.values():
        A.clean()

    if return_Ainv:
        return f, ATinv
    else:
        return f, None


Sim.fields = fields


def dask_getSourceTerm(self, tInd):
    """
    Assemble the source term. This ensures that the RHS is a vector / array
    of the correct size
    """
    source_list = self.survey.source_list
    source_block = np.array_split(source_list, cpu_count())

    def source_evaluation(simulation, sources, time):
        s_m, s_e = [], []
        for source in sources:
            sm, se = source.eval(simulation, time)
            s_m.append(sm)
            s_e.append(se)

        return s_m, s_e

    block_compute = []
    for block in source_block:
        block_compute.append(
            delayed(source_evaluation, pure=True)(self, block, self.times[tInd])
        )

    eval = dask.compute(block_compute)[0]

    s_m, s_e = [], []
    for block in eval:
        if block[0]:
            s_m.append(block[0])
            s_e.append(block[1])

    if isinstance(s_m[0][0], Zero):
        return Zero(), np.vstack(s_e).T

    return np.vstack(s_m).T, np.vstack(s_e).T


Sim.getSourceTerm = dask_getSourceTerm


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
    # ct = time()
    if f is None:
        if m is None:
            m = self.model
        f, Ainv = self.fields(m, return_Ainv=compute_J)

    # print(f"took {time() - ct} s to compute fields")
    def evaluate_receiver(source, receiver, mesh, time_mesh, fields):
        return receiver.eval(source, mesh, time_mesh, fields).flatten()

    row = delayed(evaluate_receiver, pure=True)
    rows = []
    for src in self.survey.source_list:
        for rx in src.receiver_list:
            rows.append(
                array.from_delayed(
                    row(src, rx, self.mesh, self.time_mesh, f),
                    dtype=np.float32,
                    shape=(rx.nD,),
                )
            )

    data = array.hstack(rows).compute()

    if compute_J and self._Jmatrix is None:
        Jmatrix = self.compute_J(f=f, Ainv=Ainv)
        return data, Jmatrix

    return data


Sim.dpred = dask_dpred
Sim.field_derivs = None


def block_deriv(time_index, field_type, source_list, mesh, time_mesh, fields, Jmatrix):
    """Compute derivatives for sources and receivers in a block"""
    field_len = len(fields[source_list[0], field_type, 0])
    df_duT = {src: {} for src in source_list}

    rx_count = 0
    for src in source_list:
        df_duT[src] = {rx: {} for rx in src.receiver_list}

        for rx in src.receiver_list:
            PTv = np.asarray(
                rx.getP(mesh, time_mesh, fields).todense().T
            ).reshape((field_len, time_mesh.n_faces, -1), order="F")
            derivative_fun = getattr(fields, "_{}Deriv".format(rx.projField), None)
            rx_ind = np.arange(rx_count, rx_count + rx.nD).astype(int)

            cur = derivative_fun(
                time_index,
                src,
                None,
                sp.csr_matrix(PTv[:, time_index, :]),
                adjoint=True,
            )
            df_duT[src][rx] = cur[0]
            Jmatrix[rx_ind, :] += cur[1].T

        rx_count += rx.nD

    return df_duT


def compute_field_derivs(simulation, Jmatrix, fields):
    """
    Compute the derivative of the fields
    """

    df_duT = []

    for time_index in range(simulation.nT + 1):
        df_duT.append(delayed(block_deriv, pure=True)(
            time_index,
            simulation._fieldType + "Solution",
            simulation.survey.source_list,
            simulation.mesh,
            simulation.time_mesh,
            fields,
            Jmatrix
        ))


    df_duT = dask.compute(df_duT)[0]

    return df_duT

def get_parallel_blocks(source_list: list, m_size: int, max_chunk_size: int):
    """
    Get the blocks of sources and receivers to be computed in parallel.

    Stored as a dictionary of source, receiver pairs index. The value is an array of indices
    for the rows of the sensitivity matrix.
    """
    data_block_size = np.ceil(max_chunk_size / (m_size * 8.0 * 1e-6))
    row_count = 0
    row_index = 0
    block_count = 0
    blocks = {0: {}}
    for src in source_list:
        for rx in src.receiver_list:

            indices = np.arange(rx.nD).astype(int)
            chunks = np.split(indices, int(np.ceil(len(indices)/data_block_size)))

            for ind, chunk in enumerate(chunks):
                chunk_size = len(chunk)

                # Condition to start a new block
                if (row_count + chunk_size) > (data_block_size * cpu_count() / 2):
                    row_count = 0
                    block_count += 1
                    blocks[block_count] = {}

                blocks[block_count][(src, rx, ind)] = chunk, np.arange(row_index, row_index + chunk_size).astype(int)
                row_index += chunk_size
                row_count += chunk_size

    return blocks


def get_field_deriv_block(simulation, block: dict, tInd: int, AdiagTinv, ATinv_df_duT_v: dict):
    """
    Stack the blocks of field derivatives for a given timestep and call the direct solver.
    """
    stacked_blocks = []
    indices = []
    count = 0
    for (src, rx, ind), (rx_ind, j_ind) in block.items():
        indices.append(
            np.arange(count, count + len(rx_ind))
        )
        count += len(rx_ind)
        if (src, rx, ind) not in ATinv_df_duT_v:
            # last timestep (first to be solved)
            stacked_blocks.append(
                simulation.field_derivs[tInd + 1][src][rx].toarray()[:, rx_ind]
            )

        else:
            Asubdiag = simulation.getAsubdiag(tInd + 1)
            stacked_blocks.append(
                np.asarray(
                    simulation.field_derivs[tInd + 1][src][rx][:, rx_ind]
                    - Asubdiag.T * ATinv_df_duT_v[(src, rx, ind)]
                )
            )

    solve = AdiagTinv * np.hstack(stacked_blocks)

    for (src, rx, ind), columns in zip(block, indices):
        ATinv_df_duT_v[(src, rx, ind)] = solve[:, columns]

    return ATinv_df_duT_v


def compute_rows(simulation, tInd, src, rx_ind, j_ind, ATinv_df_duT_v, f, Jmatrix, ftype):
    """
    Compute the rows of the sensitivity matrix for a given source and receiver.
    """
    dAsubdiagT_dm_v = simulation.getAsubdiagDeriv(
        tInd, f[src, ftype, tInd], ATinv_df_duT_v, adjoint=True
    )

    dRHST_dm_v = simulation.getRHSDeriv(
        tInd + 1, src, ATinv_df_duT_v, adjoint=True
    )  # on nodes of time mesh

    un_src = f[src, ftype, tInd + 1]
    # cell centered on time mesh
    dAT_dm_v = simulation.getAdiagDeriv(
        tInd, un_src, ATinv_df_duT_v, adjoint=True
    )

    Jmatrix[j_ind, :] += (-dAT_dm_v - dAsubdiagT_dm_v + dRHST_dm_v).T



def compute_J(self, f=None, Ainv=None):
    """
    Compute the rows for the sensitivity matrix.
    """

    if f is None:
        f, Ainv = self.fields(self.model, return_Ainv=True)

    ftype = self._fieldType + "Solution"
    Jmatrix = np.zeros((self.survey.nD, self.model.size), dtype=np.float32)
    blocks = get_parallel_blocks(self.survey.source_list, self.model.shape[0], self.max_chunk_size)

    self.field_derivs = compute_field_derivs(self, Jmatrix, f)

    ATinv_df_duT_v = {}
    for tInd, dt in tqdm(zip(reversed(range(self.nT)), reversed(self.time_steps))):
        AdiagTinv = Ainv[dt]
        j_row_updates = []

        for block in blocks.values():
            ATinv_df_duT_v = get_field_deriv_block(self, block, tInd, AdiagTinv, ATinv_df_duT_v)

            for (src, rx, ind), (rx_ind, j_ind) in block.items():
                j_row_updates.append(delayed(compute_rows, pure=True)(
                    self,
                    tInd,
                    src,
                    rx_ind,
                    j_ind,
                    ATinv_df_duT_v[(src, rx, ind)],
                    f,
                    Jmatrix,
                    ftype
                ))
            # for (src, rx, ind), (rx_ind, j_ind) in block.items():


        dask.compute(j_row_updates)

        # rx_count = 0
        # for isrc, src in enumerate(self.survey.source_list):
        #
        #     if isrc not in ATinv_df_duT_v:
        #         ATinv_df_duT_v[isrc] = {}
        #
        #     for rx in src.receiver_list:
        #         if rx not in ATinv_df_duT_v[isrc]:
        #             ATinv_df_duT_v[isrc][rx] = {}
        #
        #         rx_ind = np.arange(rx_count, rx_count + rx.nD).astype(int)
        #         # solve against df_duT_v
        #         if tInd >= self.nT - 1:
        #             # last timestep (first to be solved)
        #             ATinv_df_duT_v[isrc][rx] = (
        #                 AdiagTinv
        #                 * self.field_derivs[tInd+1][src][rx].toarray()
        #             )
        #         elif tInd > -1:
        #             ATinv_df_duT_v[isrc][rx] = AdiagTinv * np.asarray(
        #                 self.field_derivs[tInd+1][src][rx]
        #                 - Asubdiag.T * ATinv_df_duT_v[isrc][rx]
        #             )
        #
        #         dAsubdiagT_dm_v = self.getAsubdiagDeriv(
        #             tInd, f[src, ftype, tInd], ATinv_df_duT_v[isrc][rx], adjoint=True
        #         )
        #
        #         dRHST_dm_v = self.getRHSDeriv(
        #             tInd + 1, src, ATinv_df_duT_v[isrc][rx], adjoint=True
        #         )  # on nodes of time mesh
        #
        #         un_src = f[src, ftype, tInd + 1]
        #         # cell centered on time mesh
        #         dAT_dm_v = self.getAdiagDeriv(
        #             tInd, un_src, ATinv_df_duT_v[isrc][rx], adjoint=True
        #         )
        #
        #         Jmatrix[rx_ind, :] += (-dAT_dm_v - dAsubdiagT_dm_v + dRHST_dm_v).T

                # rx_count += rx.nD


    for A in Ainv.values():
        A.clean()

    if self.store_sensitivities == "disk":
        del Jmatrix
        return array.from_zarr(self.sensitivity_path + f"J.zarr")
    else:
        return Jmatrix


    # if f is None:
    #     f, Ainv = self.fields(self.model, return_Ainv=True)
    #
    # m_size = self.model.size
    # row_chunks = int(
    #     np.ceil(
    #         float(self.survey.nD)
    #         / np.ceil(float(m_size) * self.survey.nD * 8.0 * 1e-6 / self.max_chunk_size)
    #     )
    # )
    #
    # if self.store_sensitivities == "disk":
    #     self.J_initializer = zarr.open(
    #         self.sensitivity_path + f"J_initializer.zarr",
    #         mode="w",
    #         shape=(self.survey.nD, m_size),
    #         chunks=(row_chunks, m_size),
    #     )
    # else:
    #     self.J_initializer = np.zeros((self.survey.nD, m_size), dtype=np.float32)
    # solution_type = self._fieldType + "Solution"  # the thing we solved for
    #
    # if self.field_derivs is None:
    #     # print("Start loop for field derivs")
    #     block_size = len(f[self.survey.source_list[0], solution_type, 0])
    #
    #     field_derivs = []
    #     for tInd in range(self.nT + 1):
    #         d_count = 0
    #         df_duT_v = []
    #         for i_s, src in enumerate(self.survey.source_list):
    #             src_field_derivs = delayed(block_deriv, pure=True)(
    #                 self, src, tInd, f, block_size, d_count
    #             )
    #             df_duT_v += [src_field_derivs]
    #             d_count += np.sum([rx.nD for rx in src.receiver_list])
    #
    #         field_derivs += [df_duT_v]
    #     # print("Dask loop field derivs")
    #     # tc = time()
    #
    #     self.field_derivs = dask.compute(field_derivs)[0]
    #     # print(f"Done in {time() - tc} seconds")
    #
    # if self.store_sensitivities == "disk":
    #     Jmatrix = (
    #         zarr.open(
    #             self.sensitivity_path + f"J.zarr",
    #             mode="w",
    #             shape=(self.survey.nD, m_size),
    #             chunks=(row_chunks, m_size),
    #         )
    #         + self.J_initializer
    #     )
    # else:
    #     Jmatrix = dask.delayed(
    #         np.zeros((self.survey.nD, m_size), dtype=np.float32) + self.J_initializer
    #     )
    #
    # f = dask.delayed(f)
    # field_derivs_t = {}
    # d_block_size = np.ceil(128.0 / (m_size * 8.0 * 1e-6))
    #
    # # Check which time steps we need to compute
    # simulation_times = np.r_[0, np.cumsum(self.time_steps)] + self.t0
    # data_times = self.survey.source_list[0].receiver_list[0].times
    # n_times = len(data_times)
    #
    # for tInd, dt in tqdm(zip(reversed(range(self.nT)), reversed(self.time_steps))):
    #     AdiagTinv = Ainv[dt]
    #     Asubdiag = self.getAsubdiag(tInd)
    #     row_count = 0
    #     row_blocks = []
    #     field_derivs = {}
    #     source_blocks = []
    #     d_count = 0
    #
    #     data_bool = data_times > simulation_times[tInd]
    #
    #     if data_bool.sum() == 0:
    #         continue
    #
    #     # tc_loop = time()
    #     # print(f"Loop sources for {tInd}")
    #     for isrc, src in enumerate(self.survey.source_list):
    #
    #         column_inds = np.hstack([
    #             np.kron(np.ones(rec.locations.shape[0], dtype=bool), data_bool
    #         ) for rec in src.receiver_list])
    #
    #         if isrc not in field_derivs_t:
    #             field_derivs[(isrc, src)] = self.field_derivs[tInd + 1][isrc].toarray()[
    #                 :, column_inds
    #             ]
    #         else:
    #             field_derivs[(isrc, src)] = field_derivs_t[isrc][:, column_inds]
    #
    #         d_count += column_inds.sum()
    #
    #         if d_count > d_block_size:
    #             source_blocks = block_append(
    #                 self,
    #                 f,
    #                 AdiagTinv,
    #                 field_derivs,
    #                 m_size,
    #                 row_count,
    #                 tInd,
    #                 solution_type,
    #                 Jmatrix,
    #                 Asubdiag,
    #                 source_blocks,
    #                 data_bool,
    #             )
    #             field_derivs = {}
    #             row_count = d_count
    #             d_count = 0
    #
    #     if field_derivs:
    #         source_blocks = block_append(
    #             self,
    #             f,
    #             AdiagTinv,
    #             field_derivs,
    #             m_size,
    #             row_count,
    #             tInd,
    #             solution_type,
    #             Jmatrix,
    #             Asubdiag,
    #             source_blocks,
    #             data_bool,
    #         )
    #
    #     # print(f"Done in {time() - tc_loop} seconds")
    #     # tc = time()
    #     # print(f"Compute field derivs for {tInd}")
    #     del field_derivs_t
    #     field_derivs_t = {
    #         isrc: elem for isrc, elem in enumerate(dask.compute(source_blocks)[0])
    #     }
    #     # print(f"Done in {time() - tc} seconds")
    #
    # for A in Ainv.values():
    #     A.clean()
    #
    # if self.store_sensitivities == "disk":
    #     del Jmatrix
    #     return array.from_zarr(self.sensitivity_path + f"J.zarr")
    # else:
    #     return Jmatrix.compute()


Sim.compute_J = compute_J


def block_append(
    simulation,
    fields,
    AdiagTinv,
    field_derivs,
    m_size,
    row_count,
    tInd,
    solution_type,
    Jmatrix,
    Asubdiag,
    source_blocks,
    data_bool,
):
    solves = AdiagTinv * np.hstack(list(field_derivs.values()))
    count = 0

    for (isrc, src), block in field_derivs.items():

        column_inds = np.hstack([
            np.kron(np.ones(rec.locations.shape[0], dtype=bool), data_bool
        ) for rec in src.receiver_list])

        n_rows = column_inds.sum()
        source_blocks.append(
            dask.array.from_delayed(
                delayed(parallel_block_compute, pure=True)(
                    simulation,
                    fields,
                    src,
                    solves[:, count : count + n_rows],
                    row_count,
                    tInd,
                    solution_type,
                    Jmatrix,
                    Asubdiag,
                    simulation.field_derivs[tInd][isrc],
                    column_inds,
                ),
                shape=simulation.field_derivs[tInd + 1][isrc].shape,
                dtype=np.float32,
            )
        )
        count += n_rows
        # print(f"Appending block {isrc} in {time() - tc} seconds")
        row_count += len(column_inds)

    return source_blocks


# def block_deriv(simulation, src, tInd, f, block_size, row_count):
#     src_field_derivs = None
#     for rx in src.receiver_list:
#         v = sp.eye(rx.nD, dtype=float)
#         PT_v = rx.evalDeriv(
#             src, simulation.mesh, simulation.time_mesh, f, v, adjoint=True
#         )
#         df_duTFun = getattr(f, "_{}Deriv".format(rx.projField), None)
#
#         cur = df_duTFun(
#             simulation.nT,
#             src,
#             None,
#             PT_v[tInd * block_size: (tInd + 1) * block_size, :],
#             adjoint=True,
#         )
#
#         if not isinstance(cur[1], Zero):
#             simulation.J_initializer[row_count: row_count + rx.nD, :] += cur[1].T
#
#         if src_field_derivs is None:
#             src_field_derivs = cur[0]
#         else:
#             src_field_derivs = sp.hstack([src_field_derivs, cur[0]])
#
#         row_count += rx.nD
#
#     return src_field_derivs


def parallel_block_compute(
    simulation,
    f,
    src,
    ATinv_df_duT_v,
    row_count,
    tInd,
    solution_type,
    Jmatrix,
    Asubdiag,
    field_derivs,
    data_bool,
):
    rows = row_count + np.where(data_bool)[0]
    field_derivs_t = np.asarray(field_derivs.todense())
    field_derivs_t[:, data_bool] -= Asubdiag.T * ATinv_df_duT_v

    dAsubdiagT_dm_v = simulation.getAsubdiagDeriv(
        tInd, f[src, solution_type, tInd], ATinv_df_duT_v, adjoint=True
    )
    dRHST_dm_v = simulation.getRHSDeriv(tInd + 1, src, ATinv_df_duT_v, adjoint=True)
    un_src = f[src, solution_type, tInd + 1]
    dAT_dm_v = simulation.getAdiagDeriv(tInd, un_src, ATinv_df_duT_v, adjoint=True)
    Jmatrix[rows, :] += (-dAT_dm_v - dAsubdiagT_dm_v + dRHST_dm_v).T

    return field_derivs_t
