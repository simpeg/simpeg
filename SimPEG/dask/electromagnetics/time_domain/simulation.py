import dask
import dask.array
import os
from ....electromagnetics.time_domain.simulation import BaseTDEMSimulation as Sim
from ....utils import Zero
from multiprocessing import cpu_count
import numpy as np
import scipy.sparse as sp
from dask import array, delayed
from dask.diagnostics import ProgressBar
from SimPEG.dask.simulation import dask_Jvec, dask_Jtvec, dask_getJtJdiag
from SimPEG.dask.utils import get_parallel_blocks
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

    for tInd, dt in tqdm(enumerate(self.time_steps)):
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
    @delayed
    def evaluate_receivers(source_list, indices, mesh, time_mesh, fields):
        data = []
        for ind in indices:
            source = source_list[ind]
            for receiver in source.receiver_list:
                data.append(receiver.eval(source, mesh, time_mesh, fields).flatten())

        return np.hstack(data)

    rows = []
    fields = delayed(f)
    indices = np.array_split(np.arange(len(self.survey.source_list)), cpu_count())
    for block in indices:
        n_data = np.sum(self.survey.source_list[ind].nD for ind in block)
        rows.append(
            array.from_delayed(
                evaluate_receivers(
                    self.survey.source_list, block, self.mesh, self.time_mesh, fields
                ),
                dtype=np.float64,
                shape=(n_data,),
            )
        )

    with ProgressBar():
        data = array.hstack(rows).compute()

    if compute_J and self._Jmatrix is None:
        Jmatrix = self.compute_J(f=f, Ainv=Ainv)
        return data, Jmatrix

    return data


Sim.dpred = dask_dpred
Sim.field_derivs = None


@delayed
def delayed_block_deriv(
    time_index, chunks, field_type, source_list, mesh, time_mesh, fields, shape
):
    """Compute derivatives for sources and receivers in a block"""
    field_len = len(fields[source_list[0], field_type, 0])
    df_duT = []
    j_update = []

    for indices, arrays in chunks:
        source = source_list[indices[0]]
        receiver = source.receiver_list[indices[1]]
        PTv = receiver.getP(mesh, time_mesh, fields).tocsr()
        derivative_fun = getattr(fields, "_{}Deriv".format(receiver.projField), None)
        cur = derivative_fun(
            time_index,
            source,
            None,
            PTv[:, (time_index * field_len) : ((time_index + 1) * field_len)].T,
            adjoint=True,
        )
        df_duT.append(cur[0])

        if not isinstance(cur[1], Zero):
            j_update.append(cur[1].T)
        else:
            j_update.append(
                sp.csr_matrix((arrays[0].shape[0], shape), dtype=np.float32)
            )

    return df_duT, j_update


def compute_field_derivs(simulation, fields, blocks, Jmatrix):
    """
    Compute the derivative of the fields
    """
    df_duT = []

    for time_index in range(simulation.nT + 1):
        block_derivs = []
        block_updates = []
        delayed_chunks = []

        tc = time()
        print("Prepping blocks")
        for chunks in blocks:
            if len(chunks) == 0:
                continue

            delayed_block = delayed_block_deriv(
                time_index,
                chunks,
                simulation._fieldType + "Solution",
                simulation.survey.source_list,
                simulation.mesh,
                simulation.time_mesh,
                fields,
                simulation.model.size,
            )
            delayed_chunks.append(delayed_block)
        print(f"Done {time() - tc}")
        tc = time()
        print("Computing blocks")
        result = dask.compute(delayed_chunks)
        print(f"Done {time() - tc}")
        for chunk in result[0]:
            block_derivs.append(chunk[0])
            block_updates += chunk[1]

        j_updates = sp.vstack(block_updates)

        if len(j_updates.data) > 0:
            Jmatrix += sp.vstack(j_updates)
            if simulation.store_sensitivities == "disk":
                sens_name = simulation.sensitivity_path[:-5] + f"_{time_index % 2}.zarr"
                array.to_zarr(Jmatrix, sens_name, compute=True, overwrite=True)
                Jmatrix = array.from_zarr(sens_name)
            else:
                dask.compute(Jmatrix)

        df_duT.append(block_derivs)

    return df_duT, Jmatrix


@delayed
def deriv_block(
    s_id, r_id, b_id, ATinv_df_duT_v, Asubdiag, local_ind, sub_ind, field_derivs, tInd
):
    if (s_id, r_id, b_id) not in ATinv_df_duT_v:
        # last timestep (first to be solved)
        stacked_block = field_derivs.toarray()[:, sub_ind]

    else:
        stacked_block = np.asarray(
            field_derivs[:, sub_ind]
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
    simulation,
    block: list,
    field_derivs: list,
    tInd: int,
    AdiagTinv,
    ATinv_df_duT_v: dict,
    time_mask,
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

    for ((s_id, r_id, b_id), (rx_ind, j_ind, shape)), field_deriv in zip(
        block, field_derivs
    ):
        # Cut out early data
        time_check = np.kron(time_mask, np.ones(shape, dtype=bool))[rx_ind]
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
            field_deriv,
            tInd,
        )

        stacked_blocks.append(
            array.from_delayed(
                deriv_comp,
                dtype=float,
                shape=(
                    field_deriv.shape[0],
                    len(local_ind),
                ),
            )
        )
    if len(stacked_blocks) > 0:
        blocks = array.hstack(stacked_blocks).compute()
        solve = (AdiagTinv * blocks).reshape(blocks.shape)
    else:
        solve = None

    update_list = []
    for (address, arrays), field_deriv in zip(block, field_derivs):
        shape = (
            field_deriv.shape[0],
            len(arrays[0]),
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
    chunks,
    ATinv_df_duT_v,
    fields,
    time_mask,
):
    """
    Compute the rows of the sensitivity matrix for a given source and receiver.
    """
    n_rows = np.sum(len(chunk[1][0]) for chunk in chunks)
    rows = []

    for address, ind_array in chunks:
        src = simulation.survey.source_list[address[0]]
        time_check = np.kron(time_mask, np.ones(ind_array[2], dtype=bool))[ind_array[0]]
        local_ind = np.arange(len(ind_array[0]))[time_check]

        if len(local_ind) < 1:
            return

        field_derivs = ATinv_df_duT_v[address]
        dAsubdiagT_dm_v = simulation.getAsubdiagDeriv(
            tInd,
            fields[:, address[0], tInd],
            field_derivs[:, local_ind],
            adjoint=True,
        )

        dRHST_dm_v = simulation.getRHSDeriv(
            tInd + 1, src, field_derivs[:, local_ind], adjoint=True
        )  # on nodes of time mesh

        un_src = fields[:, address[0], tInd + 1]
        # cell centered on time mesh
        dAT_dm_v = simulation.getAdiagDeriv(
            tInd, un_src, field_derivs[:, local_ind], adjoint=True
        )
        # if isinstance(Jmatrix, zarr.core.Array):
        #     Jmatrix.oindex[ind_array[1][time_check].tolist(), :] += (-dAT_dm_v - dAsubdiagT_dm_v + dRHST_dm_v).T.astype(np.float64)
        # else:
        row_block = np.zeros(
            (len(ind_array[1]), simulation.model.size), dtype=np.float32
        )
        row_block[time_check, :] = (-dAT_dm_v - dAsubdiagT_dm_v + dRHST_dm_v).T.astype(
            np.float32
        )
        rows.append(row_block)

    return np.vstack(rows)


def compute_J(self, f=None, Ainv=None):
    """
    Compute the rows for the sensitivity matrix.
    """

    if f is None:
        f, Ainv = self.fields(self.model, return_Ainv=True)

    ftype = self._fieldType + "Solution"
    sens_name = self.sensitivity_path[:-5]
    if self.store_sensitivities == "disk":
        rows = array.zeros(
            (self.survey.nD, self.model.size),
            chunks=(self.max_chunk_size, self.model.size),
            dtype=np.float32,
        )
        Jmatrix = array.to_zarr(
            rows,
            os.path.join(sens_name + "_1.zarr"),
            compute=True,
            return_stored=True,
            overwrite=True,
        )
    else:
        Jmatrix = np.zeros((self.survey.nD, self.model.size), dtype=np.float64)

    simulation_times = np.r_[0, np.cumsum(self.time_steps)] + self.t0
    data_times = self.survey.source_list[0].receiver_list[0].times
    blocks = get_parallel_blocks(
        self.survey.source_list, self.model.shape[0], self.max_chunk_size
    )
    tc = time()
    print("Computing field derivatives")
    times_field_derivs, Jmatrix = compute_field_derivs(self, f, blocks, Jmatrix)
    print(f"Done {time() -tc}")
    fields_array = delayed(f[:, ftype, :])
    ATinv_df_duT_v = {}
    for tInd, dt in tqdm(zip(reversed(range(self.nT)), reversed(self.time_steps))):
        AdiagTinv = Ainv[dt]
        j_row_updates = []
        time_mask = data_times > simulation_times[tInd]
        tc = time()
        print("Computing derivative block")
        for block, field_deriv in zip(blocks, times_field_derivs[tInd + 1]):
            ATinv_df_duT_v = get_field_deriv_block(
                self, block, field_deriv, tInd, AdiagTinv, ATinv_df_duT_v, time_mask
            )

            if len(block) == 0:
                continue

            j_row_updates.append(
                array.from_delayed(
                    compute_rows(
                        self,
                        tInd,
                        block,
                        ATinv_df_duT_v,
                        fields_array,
                        time_mask,
                    ),
                    dtype=np.float32,
                    shape=(
                        np.sum(len(chunk[1][0]) for chunk in block),
                        self.model.size,
                    ),
                )
            )
        print(f"Done {time() - tc}")
        # Jmatrix = Jmatrix + array.vstack(j_row_updates)
        if self.store_sensitivities == "disk":
            sens_name = self.sensitivity_path[:-5] + f"_{tInd % 2}.zarr"
            array.to_zarr(
                Jmatrix + array.vstack(j_row_updates),
                sens_name,
                compute=True,
                overwrite=True,
            )
            Jmatrix = array.from_zarr(sens_name)
        else:
            tc = time()
            Jmatrix += array.vstack(j_row_updates).compute()

    for A in Ainv.values():
        A.clean()

    if self.store_sensitivities == "ram":
        return Jmatrix.compute()

    return Jmatrix


Sim.compute_J = compute_J
