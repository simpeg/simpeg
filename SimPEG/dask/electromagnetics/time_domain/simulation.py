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


@delayed
def block_deriv(time_index, field_type, source_list, mesh, time_mesh, fields, Jmatrix):
    """Compute derivatives for sources and receivers in a block"""
    field_len = len(fields[source_list[0], field_type, 0])
    df_duT = []
    rx_count = 0
    for source in source_list:
        sources_block = []

        for rx in source.receiver_list:
            PTv = rx.getP(mesh, time_mesh, fields).tocsr()
            derivative_fun = getattr(fields, "_{}Deriv".format(rx.projField), None)
            rx_ind = np.arange(rx_count, rx_count + rx.nD).astype(int)
            cur = derivative_fun(
                time_index,
                source,
                None,
                PTv[:, (time_index * field_len) : ((time_index + 1) * field_len)].T,
                adjoint=True,
            )
            sources_block.append(cur[0])

            if not isinstance(cur[1], Zero):
                Jmatrix[rx_ind, :] += cur[1].T

            rx_count += rx.nD

        df_duT.append(sources_block)

    return df_duT


def compute_field_derivs(simulation, Jmatrix, fields):
    """
    Compute the derivative of the fields
    """
    df_duT = []

    for time_index in range(simulation.nT + 1):
        df_duT.append(
            block_deriv(
                time_index,
                simulation._fieldType + "Solution",
                simulation.survey.source_list,
                simulation.mesh,
                simulation.time_mesh,
                fields,
                Jmatrix,
            )
        )

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
    for s_id, src in enumerate(source_list):
        for r_id, rx in enumerate(src.receiver_list):
            indices = np.arange(rx.nD).astype(int)
            chunks = np.array_split(
                indices, int(np.ceil(len(indices) / data_block_size))
            )

            for ind, chunk in enumerate(chunks):
                chunk_size = len(chunk)

                # Condition to start a new block
                if (row_count + chunk_size) > (data_block_size * cpu_count()):
                    row_count = 0
                    block_count += 1
                    blocks[block_count] = {}

                blocks[block_count][(s_id, r_id, ind)] = chunk, np.arange(
                    row_index, row_index + chunk_size
                ).astype(int)
                row_index += chunk_size
                row_count += chunk_size

    return blocks


@delayed
def deriv_block(
    s_id, r_id, b_id, ATinv_df_duT_v, Asubdiag, local_ind, sub_ind, simulation, tInd
):
    if (s_id, r_id, b_id) not in ATinv_df_duT_v:
        # last timestep (first to be solved)
        stacked_block = simulation.field_derivs[tInd + 1][s_id][r_id].toarray()[
            :, sub_ind
        ]

    else:
        stacked_block = np.asarray(
            simulation.field_derivs[tInd + 1][s_id][r_id][:, sub_ind]
            - Asubdiag.T * ATinv_df_duT_v[(s_id, r_id, b_id)][:, local_ind]
        )

    return stacked_block


def update_deriv_blocks(address, tInd, indices, derivatives, solve, shape):
    if address not in derivatives:
        deriv_array = np.zeros(shape)
    else:
        deriv_array = derivatives[address].compute()

    if address in indices:
        columns, local_ind = indices[address]
        deriv_array[:, local_ind] = solve[:, columns]

    derivatives[address] = delayed(deriv_array)


def get_field_deriv_block(
    simulation, block: dict, tInd: int, AdiagTinv, ATinv_df_duT_v: dict, time_mask
):
    """
    Stack the blocks of field derivatives for a given timestep and call the direct solver.
    """
    stacked_blocks = []
    indices = {}
    count = 0

    Asubdiag = None
    if tInd < simulation.nT - 1:
        Asubdiag = simulation.getAsubdiag(tInd + 1)

    for (s_id, r_id, b_id), (rx_ind, j_ind) in block.items():
        # Cut out early data
        rx = simulation.survey.source_list[s_id].receiver_list[r_id]
        time_check = np.kron(time_mask, np.ones(rx.locations.shape[0], dtype=bool))[
            rx_ind
        ]
        sub_ind = rx_ind[time_check]
        local_ind = np.arange(rx_ind.shape[0])[time_check]

        if len(sub_ind) < 1:
            continue

        indices[(s_id, r_id, b_id)] = (
            np.arange(count, count + len(sub_ind)),
            local_ind,
        )
        count += len(sub_ind)
        deriv_comp = deriv_block(
            s_id,
            r_id,
            b_id,
            ATinv_df_duT_v,
            Asubdiag,
            local_ind,
            sub_ind,
            simulation,
            tInd,
        )

        stacked_blocks.append(
            array.from_delayed(
                deriv_comp,
                dtype=float,
                shape=(
                    simulation.field_derivs[tInd][s_id][r_id].shape[0],
                    len(local_ind),
                ),
            )
        )
    if len(stacked_blocks) > 0:
        blocks = array.hstack(stacked_blocks).compute()
        solve = AdiagTinv * blocks
    else:
        solve = None

    update_list = []
    for address in block:
        shape = (
            simulation.field_derivs[tInd][address[0]][address[1]].shape[0],
            len(block[address][0]),
        )
        update_list.append(
            update_deriv_blocks(address, tInd, indices, ATinv_df_duT_v, solve, shape)
        )
    dask.compute(update_list)

    return ATinv_df_duT_v


@delayed
def compute_rows(
    simulation,
    tInd,
    address,  # (s_id, r_id, b_id)
    indices,  # (rx_ind, j_ind),
    ATinv_df_duT_v,
    fields,
    Jmatrix,
    ftype,
    time_mask,
):
    """
    Compute the rows of the sensitivity matrix for a given source and receiver.
    """
    src = simulation.survey.source_list[address[0]]
    rx = src.receiver_list[address[1]]
    time_check = np.kron(time_mask, np.ones(rx.locations.shape[0], dtype=bool))[
        indices[0]
    ]
    local_ind = np.arange(indices[0].shape[0])[time_check]

    if len(local_ind) < 1:
        return

    dAsubdiagT_dm_v = simulation.getAsubdiagDeriv(
        tInd,
        fields[src, ftype, tInd],
        ATinv_df_duT_v[address][:, local_ind],
        adjoint=True,
    )

    dRHST_dm_v = simulation.getRHSDeriv(
        tInd + 1, src, ATinv_df_duT_v[address][:, local_ind], adjoint=True
    )  # on nodes of time mesh

    un_src = fields[src, ftype, tInd + 1]
    # cell centered on time mesh
    dAT_dm_v = simulation.getAdiagDeriv(
        tInd, un_src, ATinv_df_duT_v[address][:, local_ind], adjoint=True
    )

    Jmatrix[indices[1][time_check], :] += (-dAT_dm_v - dAsubdiagT_dm_v + dRHST_dm_v).T


def compute_J(self, f=None, Ainv=None):
    """
    Compute the rows for the sensitivity matrix.
    """

    if f is None:
        f, Ainv = self.fields(self.model, return_Ainv=True)

    ftype = self._fieldType + "Solution"
    Jmatrix = delayed(np.zeros((self.survey.nD, self.model.size), dtype=np.float32))
    simulation_times = np.r_[0, np.cumsum(self.time_steps)] + self.t0
    data_times = self.survey.source_list[0].receiver_list[0].times
    blocks = get_parallel_blocks(
        self.survey.source_list, self.model.shape[0], self.max_chunk_size
    )
    self.field_derivs = compute_field_derivs(self, Jmatrix, f)
    ATinv_df_duT_v = {}
    for tInd, dt in tqdm(zip(reversed(range(self.nT)), reversed(self.time_steps))):
        AdiagTinv = Ainv[dt]
        j_row_updates = []
        time_mask = data_times > simulation_times[tInd]

        for block in blocks.values():
            ATinv_df_duT_v = get_field_deriv_block(
                self, block, tInd, AdiagTinv, ATinv_df_duT_v, time_mask
            )

            for address, indices in block.items():
                j_row_updates.append(
                    compute_rows(
                        self,
                        tInd,
                        address,
                        indices,
                        ATinv_df_duT_v,
                        f,
                        Jmatrix,
                        ftype,
                        time_mask,
                    )
                )
        dask.compute(j_row_updates)
    for A in Ainv.values():
        A.clean()

    if self.store_sensitivities == "disk":
        del Jmatrix
        return array.from_zarr(self.sensitivity_path + f"J.zarr")
    else:
        return Jmatrix.compute()


Sim.compute_J = compute_J
