import dask
import dask.array
import os
from ....electromagnetics.time_domain.simulation import BaseTDEMSimulation as Sim

from ....utils import Zero
from ...simulation import getJtJdiag, Jvec, Jtvec, Jmatrix

import numpy as np
import scipy.sparse as sp
from dask import array, delayed
from dask.distributed import get_client

from simpeg.dask.utils import get_parallel_blocks
from simpeg.utils import mkvc


def fields(self, m=None, return_Ainv=False):
    if m is not None:
        self.model = m

    if getattr(self, "_stashed_fields", None) is not None and not return_Ainv:
        return self._stashed_fields

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

        sol = Ainv[dt] * np.asarray(rhs)
        f[:, self._fieldType + "Solution", tInd + 1] = sol

    self._stashed_fields = f
    if return_Ainv:
        return f, Ainv
    return f


def getSourceTerm(self, tInd):
    """
    Assemble the source term. This ensures that the RHS is a vector / array
    of the correct size
    """
    if (
        getattr(self, "_stashed_sources", None) is not None
        and tInd in self._stashed_sources
    ):
        return self._stashed_sources[tInd]
    elif getattr(self, "_stashed_sources", None) is None:
        self._stashed_sources = {}

    try:
        client = get_client()
        sim = client.scatter(self, workers=self.worker)
    except ValueError:
        client = None
        sim = self

    source_list = self.survey.source_list
    source_block = np.array_split(
        np.arange(len(source_list)), self.n_threads(client=client)
    )

    if client:
        sim = client.scatter(self, workers=self.worker)
        source_list = client.scatter(source_list, workers=self.worker)
    else:
        delayed_source_eval = delayed(source_evaluation)
        sim = self

    block_compute = []
    for block in source_block:
        if client:
            block_compute.append(
                client.submit(
                    source_evaluation,
                    sim,
                    block,
                    self.times[tInd],
                    source_list,
                    workers=self.worker,
                )
            )
        else:
            block_compute.append(
                delayed_source_eval(self, block, self.times[tInd], source_list)
            )

    if client:
        blocks = client.gather(block_compute)
    else:
        blocks = dask.compute(block_compute)[0]

    s_m, s_e = [], []
    for block in blocks:
        if block[0]:
            s_m.append(block[0])
            s_e.append(block[1])

    if isinstance(s_m[0][0], Zero):
        self._stashed_sources[tInd] = Zero(), sp.csr_matrix(np.vstack(s_e).T)
    else:
        self._stashed_sources[tInd] = sp.csr_matrix(np.vstack(s_m).T), sp.csr_matrix(
            np.vstack(s_e).T
        )

    return self._stashed_sources[tInd]


def compute_J(self, m, f=None):
    """
    Compute the rows for the sensitivity matrix.
    """
    if f is None:
        f, Ainv = self.fields(m=m, return_Ainv=True)

    try:
        client = get_client()
    except ValueError:
        client = None

    ftype = self._fieldType + "Solution"
    sens_name = self.sensitivity_path[:-5]
    if self.store_sensitivities == "disk":
        rows = array.zeros(
            (self.survey.nD, m.size),
            chunks=(self.max_chunk_size, m.size),
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
        Jmatrix = np.zeros((self.survey.nD, m.size), dtype=np.float64)

    simulation_times = np.r_[0, np.cumsum(self.time_steps)] + self.t0
    data_times = self.survey.source_list[0].receiver_list[0].times
    compute_row_size = np.ceil(self.max_chunk_size / (m.shape[0] * 8.0 * 1e-6))
    blocks = get_parallel_blocks(
        self.survey.source_list,
        compute_row_size,
        thread_count=self.n_threads(client=client),
    )
    fields_array = f[:, ftype, :]

    if len(self.survey.source_list) == 1:
        fields_array = fields_array[:, np.newaxis, :]

    times_field_derivs, Jmatrix = compute_field_derivs(
        self, f, blocks, Jmatrix, fields_array.shape, client
    )

    ATinv_df_duT_v = [[] for _ in blocks]

    if client:
        fields_array = client.scatter(fields_array, workers=self.worker)
        sim = client.scatter(self, workers=self.worker)
    else:
        delayed_compute_rows = delayed(compute_rows)
        sim = self
    for tInd, dt in zip(reversed(range(self.nT)), reversed(self.time_steps)):
        AdiagTinv = Ainv[dt]
        j_row_updates = []
        time_mask = data_times > simulation_times[tInd]

        if not np.any(time_mask):
            continue

        for ind, (block, field_deriv) in enumerate(
            zip(blocks, times_field_derivs[tInd + 1], strict=True)
        ):
            ATinv_df_duT_v[ind] = get_field_deriv_block(
                self,
                block,
                field_deriv,
                tInd,
                AdiagTinv,
                ATinv_df_duT_v[ind],
                time_mask,
                client,
            )

            if len(block) == 0:
                continue

            if client:
                field_derivatives = client.scatter(
                    ATinv_df_duT_v[ind], workers=self.worker
                )
                j_row_updates.append(
                    client.submit(
                        compute_rows,
                        sim,
                        tInd,
                        block,
                        field_derivatives,
                        fields_array,
                        time_mask,
                        workers=self.worker,
                    )
                )
            else:
                j_row_updates.append(
                    array.from_delayed(
                        delayed_compute_rows(
                            sim,
                            tInd,
                            block,
                            ATinv_df_duT_v[ind],
                            fields_array,
                            time_mask,
                        ),
                        dtype=np.float32,
                        shape=(
                            np.sum([len(chunk[1][0]) for chunk in block]),
                            m.size,
                        ),
                    )
                )

        if client:
            j_row_updates = np.vstack(client.gather(j_row_updates))
        else:
            j_row_updates = array.vstack(j_row_updates).compute()

        if self.store_sensitivities == "disk":
            sens_name = self.sensitivity_path[:-5] + f"_{tInd % 2}.zarr"
            array.to_zarr(
                Jmatrix + j_row_updates,
                sens_name,
                compute=True,
                overwrite=True,
            )
            Jmatrix = array.from_zarr(sens_name)
        else:
            Jmatrix += j_row_updates

    for A in Ainv.values():
        A.clean()

    if self.store_sensitivities == "ram":
        self._Jmatrix = np.asarray(Jmatrix)

    self._Jmatrix = Jmatrix

    return self._Jmatrix


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


def source_evaluation(simulation, indices, time_channel, sources):
    s_m, s_e = [], []
    for ind in indices:
        sm, se = sources[ind].eval(simulation, time_channel)
        s_m.append(sm)
        s_e.append(se)

    return s_m, s_e


def evaluate_receivers(block, mesh, time_mesh, fields, fields_array):
    data = []
    for _, ind, receiver in block:
        Ps = receiver.getSpatialP(mesh, fields)
        Pt = receiver.getTimeP(time_mesh, fields)
        vector = (Pt * (Ps * fields_array[:, ind, :]).T).flatten()

        data.append(vector)

    return np.hstack(data)


def compute_field_derivs(self, fields, blocks, Jmatrix, fields_shape, client):
    """
    Compute the derivative of the fields
    """
    delayed_chunks = []

    if client:
        mesh = client.scatter(self.mesh, workers=self.worker)
        time_mesh = client.scatter(self.time_mesh, workers=self.worker)
        fields = client.scatter(fields, workers=self.worker)
        source_list = client.scatter(self.survey.source_list, workers=self.worker)
    else:
        mesh = self.mesh
        time_mesh = self.time_mesh
        delayed_block_deriv = delayed(block_deriv)
        source_list = self.survey.source_list

    for chunks in blocks:
        if len(chunks) == 0:
            continue

        if client:
            delayed_chunks.append(
                client.submit(
                    block_deriv,
                    self.nT,
                    chunks,
                    fields_shape[0],
                    source_list,
                    mesh,
                    time_mesh,
                    fields,
                    self.model.size,
                    workers=self.worker,
                )
            )
        else:
            delayed_chunks.append(
                delayed_block_deriv(
                    self.nT,
                    chunks,
                    fields_shape[0],
                    source_list,
                    self.mesh,
                    self.time_mesh,
                    fields,
                    self.model.size,
                )
            )

    if client:
        result = client.gather(delayed_chunks)
    else:
        result = dask.compute(delayed_chunks)[0]

    df_duT = [
        [[[] for _ in block] for block in blocks if len(block) > 0]
        for _ in range(self.nT + 1)
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
        if self.store_sensitivities == "disk":
            sens_name = self.sensitivity_path[:-5] + f"_{time() % 2}.zarr"
            array.to_zarr(Jmatrix, sens_name, compute=True, overwrite=True)
            Jmatrix = array.from_zarr(sens_name)

    return df_duT, Jmatrix


def get_field_deriv_block(
    self,
    block: list,
    field_derivs: list,
    tInd: int,
    AdiagTinv,
    ATinv_df_duT_v,
    time_mask,
    client,
):
    """
    Stack the blocks of field derivatives for a given timestep and call the direct solver.
    """
    stacked_blocks = []
    if len(ATinv_df_duT_v) == 0:
        ATinv_df_duT_v = [[] for _ in block]
    indices = []
    count = 0

    Asubdiag = None
    if tInd < self.nT - 1:
        Asubdiag = self.getAsubdiag(tInd + 1)

    for (_, (rx_ind, _, shape)), field_deriv, ATinv_chunk in zip(
        block, field_derivs, ATinv_df_duT_v
    ):
        # Cut out early data
        time_check = np.kron(time_mask, np.ones(shape, dtype=bool))[rx_ind]
        local_ind = np.arange(rx_ind.shape[0])[time_check]

        if len(local_ind) < 1:
            continue

        indices.append(
            (np.arange(count, count + len(local_ind)), local_ind),
        )
        count += len(local_ind)

        if len(ATinv_chunk) == 0:
            # last timestep (first to be solved)
            stacked_block = field_deriv.toarray()[:, local_ind]

        else:
            stacked_block = np.asarray(
                field_deriv[:, local_ind] - Asubdiag.T * ATinv_chunk[:, local_ind]
            )

        stacked_blocks.append(stacked_block)

    if len(stacked_blocks) > 0:
        blocks = np.hstack(stacked_blocks)

        solve = (AdiagTinv * blocks).reshape(blocks.shape)
    else:
        solve = None

    updated_ATinv_df_duT_v = []
    for (_, arrays), field_deriv, ATinv_chunk, (columns, local_ind) in zip(
        block, field_derivs, ATinv_df_duT_v, indices, strict=True
    ):

        if len(ATinv_chunk) == 0:
            shape = (
                field_deriv.shape[0],
                len(arrays[0]),
            )
            ATinv_chunk = np.zeros(shape, dtype=np.float32)

        if solve is None:
            continue

        ATinv_chunk[:, local_ind] = solve[:, columns]
        updated_ATinv_df_duT_v.append(ATinv_chunk)

    return updated_ATinv_df_duT_v


def block_deriv(
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


def deriv_block(ATinv_df_duT_v, Asubdiag, local_ind, field_derivs):
    if len(ATinv_df_duT_v) == 0:
        # last timestep (first to be solved)
        stacked_block = field_derivs.toarray()[:, local_ind]

    else:
        stacked_block = np.asarray(
            field_derivs[:, local_ind] - Asubdiag.T * ATinv_df_duT_v[:, local_ind]
        )

    return stacked_block


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

    for (address, ind_array), field_derivs in zip(chunks, ATinv_df_duT_v):
        src = simulation.survey.source_list[address[0]]
        time_check = np.kron(time_mask, np.ones(ind_array[2], dtype=bool))[ind_array[0]]
        local_ind = np.arange(len(ind_array[0]))[time_check]

        if len(local_ind) < 1:
            row_block = np.zeros(
                (len(ind_array[1]), simulation.model.size), dtype=np.float32
            )
            rows.append(row_block)
            continue

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


Sim.fields = fields
Sim.getSourceTerm = getSourceTerm
Sim.compute_J = compute_J
Sim.getJtJdiag = getJtJdiag
Sim.Jvec = Jvec
Sim.Jtvec = Jtvec
Sim.Jmatrix = Jmatrix
