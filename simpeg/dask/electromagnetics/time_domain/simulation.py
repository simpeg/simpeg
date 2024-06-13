import dask
import dask.array
import os
from ....electromagnetics.time_domain.simulation import BaseTDEMSimulation as Sim
from ....utils import Zero
from simpeg.fields import TimeFields
from multiprocessing import cpu_count
import numpy as np
import scipy.sparse as sp
from dask import array, delayed

from simpeg.dask.simulation import dask_Jvec, dask_Jtvec, dask_getJtJdiag
from simpeg.dask.utils import get_parallel_blocks
from simpeg.utils import mkvc

from time import time
from tqdm import tqdm

Sim.sensitivity_path = "./sensitivity/"
Sim.getJtJdiag = dask_getJtJdiag
Sim.Jvec = dask_Jvec
Sim.Jtvec = dask_Jtvec
Sim.clean_on_model_update = ["_Jmatrix", "_jtjdiag"]


@delayed
def field_projection(field_array, src_list, array_ind, time_ind, func):
    fieldI = field_array[:, :, array_ind]
    if fieldI.shape[0] == fieldI.size:
        fieldI = mkvc(fieldI, 2)
    new_array = func(fieldI, src_list, time_ind)
    if new_array.ndim == 1:
        new_array = new_array[:, np.newaxis, np.newaxis]
    elif new_array.ndim == 2:
        new_array = new_array[:, :, np.newaxis]

    return new_array


def _getField(self, name, ind, src_list):
    srcInd, timeInd = ind

    if name in self._fields:
        out = self._fields[name][:, srcInd, timeInd]
    else:
        # Aliased fields
        alias, loc, func = self.aliasFields[name]
        if isinstance(func, str):
            assert hasattr(self, func), (
                "The alias field function is a string, but it does "
                "not exist in the Fields class."
            )
            func = getattr(self, func)
        pointerFields = self._fields[alias][:, srcInd, timeInd]
        pointerShape = self._correctShape(alias, ind)
        pointerFields = pointerFields.reshape(pointerShape, order="F")

        # First try to return the function as three arguments (without timeInd)
        if timeInd == slice(None, None, None):
            try:
                # assume it will take care of integrating over all times
                return func(pointerFields, srcInd)
            except TypeError:
                pass

        timeII = np.arange(self.simulation.nT + 1)[timeInd]
        if not isinstance(src_list, list):
            src_list = [src_list]

        if timeII.size == 1:
            pointerShapeDeflated = self._correctShape(alias, ind, deflate=True)
            pointerFields = pointerFields.reshape(pointerShapeDeflated, order="F")
            out = func(pointerFields, src_list, timeII)
        else:  # loop over the time steps
            arrays = []

            for i, TIND_i in enumerate(timeII):  # Need to parallelize this
                arrays.append(
                    array.from_delayed(
                        field_projection(pointerFields, src_list, i, TIND_i, func),
                        dtype=np.float32,
                        shape=(pointerShape[0], pointerShape[1], 1),
                    )
                )

            out = array.dstack(arrays).compute()

    shape = self._correctShape(name, ind, deflate=True)
    return out.reshape(shape, order="F")


TimeFields._getField = _getField


def fields(self, m=None, return_Ainv=False):
    if m is not None:
        self.model = m

    f = self.fieldsPair(self)
    f[:, self._fieldType + "Solution", 0] = self.getInitialFields()
    Ainv = {}

    for tInd, dt in enumerate(self.time_steps):
        if dt not in Ainv:
            A = self.getAdiag(tInd)
            Ainv[dt] = self.solver(sp.csr_matrix(A), **self.solver_opts)

        Asubdiag = self.getAsubdiag(tInd)
        rhs = -Asubdiag * f[:, (self._fieldType + "Solution"), tInd]

        if (
            np.abs(self.survey.source_list[0].waveform.eval(self.times[tInd + 1]))
            > 1e-8
        ):
            rhs += self.getRHS(tInd + 1)

        sol = Ainv[dt] * rhs
        f[:, self._fieldType + "Solution", tInd + 1] = sol

    if return_Ainv:
        self.Ainv = Ainv

    return f


Sim.fields = fields


@delayed
def source_evaluation(simulation, sources, time_channel):
    s_m, s_e = [], []
    for source in sources:
        sm, se = source.eval(simulation, time_channel)
        s_m.append(sm)
        s_e.append(se)

    return s_m, s_e


def dask_getSourceTerm(self, tInd):
    """
    Assemble the source term. This ensures that the RHS is a vector / array
    of the correct size
    """
    source_list = self.survey.source_list
    source_block = np.array_split(source_list, cpu_count())

    block_compute = []
    for block in source_block:
        block_compute.append(source_evaluation(self, block, self.times[tInd]))

    blocks = dask.compute(block_compute)[0]

    s_m, s_e = [], []
    for block in blocks:
        if block[0]:
            s_m.append(block[0])
            s_e.append(block[1])

    if isinstance(s_m[0][0], Zero):
        return Zero(), np.vstack(s_e).T

    return np.vstack(s_m).T, np.vstack(s_e).T


Sim.getSourceTerm = dask_getSourceTerm


@delayed
def evaluate_receivers(block, mesh, time_mesh, fields, fields_array):
    data = []
    for _, ind, receiver in block:
        Ps = receiver.getSpatialP(mesh, fields)
        Pt = receiver.getTimeP(time_mesh, fields)
        vector = (Pt * (Ps * fields_array[:, ind, :]).T).flatten()

        data.append(vector)

    return np.hstack(data)


def dask_dpred(self, m=None, f=None, compute_J=False):
    r"""
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
        f = self.fields(m, return_Ainv=compute_J)

    rows = []
    receiver_projection = self.survey.source_list[0].receiver_list[0].projField
    fields_array = f[:, receiver_projection, :]

    if len(self.survey.source_list) == 1:
        fields_array = fields_array[:, np.newaxis, :]

    all_receivers = []

    for ind, src in enumerate(self.survey.source_list):
        for rx in src.receiver_list:
            all_receivers.append((src, ind, rx))

    receiver_blocks = np.array_split(all_receivers, cpu_count())

    for block in receiver_blocks:
        n_data = np.sum([rec.nD for _, _, rec in block])
        if n_data == 0:
            continue

        rows.append(
            array.from_delayed(
                evaluate_receivers(block, self.mesh, self.time_mesh, f, fields_array),
                dtype=np.float64,
                shape=(n_data,),
            )
        )

    data = array.hstack(rows).compute()

    if compute_J and self._Jmatrix is None:
        Jmatrix = self.compute_J(f=f)
        return data, Jmatrix

    return data


Sim.dpred = dask_dpred
Sim.field_derivs = None


@delayed
def delayed_block_deriv(
    n_times, chunks, field_len, source_list, mesh, time_mesh, fields, shape
):
    """Compute derivatives for sources and receivers in a block"""
    df_duT = []
    j_updates = []

    for indices, arrays in chunks:
        j_update = 0.0
        source = source_list[indices[0]]
        receiver = source.receiver_list[indices[1]]

        spatialP = receiver.getSpatialP(mesh, fields)
        timeP = receiver.getTimeP(time_mesh, fields)

        derivative_fun = getattr(fields, "_{}Deriv".format(receiver.projField), None)
        time_derivs = []
        for time_index in range(n_times + 1):
            if len(timeP[:, time_index].data) == 0:
                time_derivs.append(
                    sp.csr_matrix((field_len, len(arrays[0])), dtype=np.float32)
                )
                j_update += sp.csr_matrix((arrays[0].shape[0], shape), dtype=np.float32)
                continue

            projection = sp.kron(timeP[:, time_index], spatialP, format="csr")
            cur = derivative_fun(
                time_index,
                source,
                None,
                projection.T,
                adjoint=True,
            )

            time_derivs.append(cur[0][:, arrays[0]])

            if not isinstance(cur[1], Zero):
                j_update += cur[1].T
            else:
                j_update += sp.csr_matrix((arrays[0].shape[0], shape), dtype=np.float32)

        j_updates.append(j_update)
        df_duT.append(time_derivs)

    return df_duT, j_updates


def compute_field_derivs(simulation, fields, blocks, Jmatrix, fields_shape):
    """
    Compute the derivative of the fields
    """
    delayed_chunks = []
    for chunks in blocks:
        if len(chunks) == 0:
            continue

        delayed_block = delayed_block_deriv(
            simulation.nT,
            chunks,
            fields_shape[0],
            simulation.survey.source_list,
            simulation.mesh,
            simulation.time_mesh,
            fields,
            simulation.model.size,
        )
        delayed_chunks.append(delayed_block)

    result = dask.compute(delayed_chunks)[0]
    df_duT = [
        [[[] for _ in block] for block in blocks if len(block) > 0]
        for _ in range(simulation.nT + 1)
    ]
    j_updates = []

    for bb, block in enumerate(result):
        j_updates += block[1]
        for cc, chunk in enumerate(block[0]):
            for ind, time_block in enumerate(chunk):
                df_duT[ind][bb][cc] = time_block

    j_updates = sp.vstack(j_updates)

    if len(j_updates.data) > 0:
        Jmatrix += j_updates
        if simulation.store_sensitivities == "disk":
            sens_name = simulation.sensitivity_path[:-5] + f"_{time() % 2}.zarr"
            array.to_zarr(Jmatrix, sens_name, compute=True, overwrite=True)
            Jmatrix = array.from_zarr(sens_name)

    return df_duT, Jmatrix


@delayed
def deriv_block(
    s_id, r_id, b_id, ATinv_df_duT_v, Asubdiag, local_ind, field_derivs, tInd
):
    if (s_id, r_id, b_id) not in ATinv_df_duT_v:
        # last timestep (first to be solved)
        stacked_block = field_derivs.toarray()[:, local_ind]

    else:
        stacked_block = np.asarray(
            field_derivs[:, local_ind]
            - Asubdiag.T * ATinv_df_duT_v[(s_id, r_id, b_id)][:, local_ind]
        )

    return stacked_block


def update_deriv_blocks(address, indices, derivatives, solve, shape):
    if address not in derivatives:
        deriv_array = np.zeros(shape)
    else:
        deriv_array = derivatives[address]

    if address in indices:
        columns, local_ind = indices[address]
        if solve is not None:
            deriv_array[:, local_ind] = solve[:, columns]

    derivatives[address] = deriv_array


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

    for ((s_id, r_id, b_id), (rx_ind, _, shape)), field_deriv in zip(
        block, field_derivs
    ):
        # Cut out early data
        time_check = np.kron(time_mask, np.ones(shape, dtype=bool))[rx_ind]
        local_ind = np.arange(rx_ind.shape[0])[time_check]

        if len(local_ind) < 1:
            continue

        indices[(s_id, r_id, b_id)] = (
            np.arange(count, count + len(local_ind)),
            local_ind,
        )
        count += len(local_ind)
        deriv_comp = deriv_block(
            s_id,
            r_id,
            b_id,
            ATinv_df_duT_v,
            Asubdiag,
            local_ind,
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

    for (address, arrays), field_deriv in zip(block, field_derivs):
        shape = (
            field_deriv.shape[0],
            len(arrays[0]),
        )

        update_deriv_blocks(address, indices, ATinv_df_duT_v, solve, shape)

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
    rows = []

    for address, ind_array in chunks:
        src = simulation.survey.source_list[address[0]]
        time_check = np.kron(time_mask, np.ones(ind_array[2], dtype=bool))[ind_array[0]]
        local_ind = np.arange(len(ind_array[0]))[time_check]

        if len(local_ind) < 1:
            row_block = np.zeros(
                (len(ind_array[1]), simulation.model.size), dtype=np.float32
            )
            rows.append(row_block)
            continue

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
        row_block = np.zeros(
            (len(ind_array[1]), simulation.model.size), dtype=np.float32
        )
        row_block[time_check, :] = (-dAT_dm_v - dAsubdiagT_dm_v + dRHST_dm_v).T.astype(
            np.float32
        )
        rows.append(row_block)

    return np.vstack(rows)


def compute_J(self, f=None):
    """
    Compute the rows for the sensitivity matrix.
    """
    if f is None:
        f = self.fields(self.model, return_Ainv=True)

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
    compute_row_size = np.ceil(self.max_chunk_size / (self.model.shape[0] * 8.0 * 1e-6))
    blocks = get_parallel_blocks(self.survey.source_list, compute_row_size)
    fields_array = f[:, ftype, :]

    if len(self.survey.source_list) == 1:
        fields_array = fields_array[:, np.newaxis, :]

    times_field_derivs, Jmatrix = compute_field_derivs(
        self, f, blocks, Jmatrix, fields_array.shape
    )

    ATinv_df_duT_v = {}
    for tInd, dt in tqdm(zip(reversed(range(self.nT)), reversed(self.time_steps))):
        AdiagTinv = self.Ainv[dt]
        j_row_updates = []
        time_mask = data_times > simulation_times[tInd]

        if not np.any(time_mask):
            continue

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
                        np.sum([len(chunk[1][0]) for chunk in block]),
                        self.model.size,
                    ),
                )
            )

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
            Jmatrix += array.vstack(j_row_updates).compute()

    for A in self.Ainv.values():
        A.clean()

    if self.store_sensitivities == "ram":
        return np.asarray(Jmatrix)

    return Jmatrix


Sim.compute_J = compute_J
