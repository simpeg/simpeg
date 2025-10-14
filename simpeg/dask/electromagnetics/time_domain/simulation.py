import os
import shutil
from ....electromagnetics.time_domain.simulation import BaseTDEMSimulation as Sim

from ....utils import Zero
from ...simulation import getJtJdiag, Jvec, Jtvec, Jmatrix

import numpy as np
import scipy.sparse as sp
from dask import array, delayed, compute
import zarr

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

    blocks = []
    for source in self.survey.source_list:
        blocks.append(source_evaluation(self, self.times[tInd], source))

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

    client, worker = self._get_client_worker()

    ftype = self._fieldType + "Solution"
    n_cells = m.size

    simulation_times = np.r_[0, np.cumsum(self.time_steps)] + self.t0
    data_times = self.survey.source_list[0].receiver_list[0].times
    compute_row_size = np.ceil(self.max_chunk_size / (m.shape[0] * 8.0 * 1e-6))
    blocks = get_parallel_blocks(
        self.survey.source_list,
        compute_row_size,
        thread_count=self.n_threads(client=client, worker=worker),
    )
    fields_array = f[:, ftype, :]

    if len(self.survey.source_list) == 1:
        fields_array = fields_array[:, np.newaxis, :]

    if self.store_sensitivities == "disk":
        chunk_size = np.median(
            [np.sum([len(chunk[1][1]) for chunk in block]) for block in blocks]
        ).astype(int)
        if os.path.exists(self.sensitivity_path):
            shutil.rmtree(self.sensitivity_path)

        Jmatrix = zarr.open(
            self.sensitivity_path,
            mode="w",
            shape=(self.survey.nD, n_cells),
            chunks=(chunk_size, n_cells),
        )
    else:
        Jmatrix = np.zeros((self.survey.nD, n_cells), dtype=np.float64)

        if client:
            Jmatrix = client.scatter(Jmatrix, workers=worker)

    times_field_derivs = compute_field_derivs(
        self, f, blocks, Jmatrix, fields_array.shape
    )

    ATinv_df_duT_v = [[] for _ in blocks]

    if client:
        fields_array = client.scatter(fields_array, workers=worker)
        sim = client.scatter(self, workers=worker)
    else:
        delayed_compute_rows = delayed(compute_rows)
        sim = self
    for tInd, dt in zip(reversed(range(self.nT)), reversed(self.time_steps)):

        AdiagTinv = Ainv[dt]
        future_updates = []
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

        if client:
            field_derivatives = client.scatter(ATinv_df_duT_v, workers=worker)
        else:
            field_derivatives = ATinv_df_duT_v

        for block_ind in range(len(blocks)):

            if len(block) == 0:
                continue

            if client:
                future_updates.append(
                    client.submit(
                        compute_rows,
                        sim,
                        tInd,
                        block_ind,
                        blocks,
                        field_derivatives,
                        fields_array,
                        time_mask,
                        Jmatrix,
                        workers=worker,
                    )
                )
            else:
                future_updates.append(
                    array.from_delayed(
                        delayed_compute_rows(
                            sim,
                            tInd,
                            block_ind,
                            blocks,
                            field_derivatives,
                            fields_array,
                            time_mask,
                            Jmatrix,
                        ),
                        dtype=np.float32,
                        shape=(
                            np.sum([len(chunk[1][0]) for chunk in block]),
                            m.size,
                        ),
                    )
                )

        if client:
            client.gather(future_updates)
        else:
            compute(future_updates)

    for A in Ainv.values():
        A.clean()

    if self.store_sensitivities == "disk":
        del Jmatrix
        self._Jmatrix = array.from_zarr(self.sensitivity_path)
    else:
        if client:
            Jmatrix = client.gather(Jmatrix)

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


def source_evaluation(simulation, time_channel, source):
    s_m, s_e = [], []

    sm, se = source.eval(simulation, time_channel)
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


def compute_field_derivs(self, fields, blocks, Jmatrix, fields_shape):
    """
    Compute the derivative of the fields
    """
    delayed_chunks = []

    client, worker = self._get_client_worker()

    if client:
        mesh = client.scatter(self.mesh, workers=worker)
        time_mesh = client.scatter(self.time_mesh, workers=worker)
        fields = client.scatter(fields, workers=worker)
        source_list = client.scatter(self.survey.source_list, workers=worker)
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
                    Jmatrix,
                    workers=worker,
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
                    Jmatrix,
                )
            )

    if client:
        result = client.gather(delayed_chunks)
    else:
        result = compute(delayed_chunks)[0]

    df_duT = [[[[] for _ in block] for block in blocks] for _ in range(self.nT + 1)]

    for bb, block in enumerate(result):
        for cc, chunk in enumerate(block):
            for ind, time_block in enumerate(chunk):
                df_duT[ind][bb][cc] = time_block

    return df_duT


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
    if len(ATinv_df_duT_v) == 0:
        ATinv_df_duT_v = [[] for _ in block]

    Asubdiag = None
    if tInd < self.nT - 1:
        Asubdiag = self.getAsubdiag(tInd + 1)

    updated_ATinv_df_duT_v = []

    for (_, (rx_ind, _, shape)), field_deriv, ATinv_chunk in zip(
        block, field_derivs, ATinv_df_duT_v
    ):

        # Cut out early data
        time_check = np.kron(time_mask, np.ones(shape, dtype=bool))[rx_ind]
        local_ind = np.arange(rx_ind.shape[0])[time_check]

        if len(ATinv_chunk) == 0:
            # last timestep (first to be solved)
            time_block = field_deriv.toarray()[:, local_ind]
            shape = (
                field_deriv.shape[0],
                len(rx_ind),
            )
            ATinv_chunk = np.zeros(shape, dtype=np.float32)
        else:
            time_block = np.asarray(
                field_deriv[:, local_ind] - Asubdiag.T * ATinv_chunk[:, local_ind]
            )

        if time_block.ndim == 2 and time_block.shape[1] > 0:
            solve = (AdiagTinv * time_block).reshape(time_block.shape)
            ATinv_chunk[:, local_ind] = solve

        updated_ATinv_df_duT_v.append(ATinv_chunk)

    return updated_ATinv_df_duT_v


def block_deriv(
    n_times, chunks, field_len, source_list, mesh, time_mesh, fields, shape, Jmatrix
):
    """Compute derivatives for sources and receivers in a block"""
    df_duT = []
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

        if isinstance(Jmatrix, zarr.Array):
            j_slice = Jmatrix.get_orthogonal_selection((arrays[1], slice(None)))

            Jmatrix.set_orthogonal_selection(
                (arrays[1], slice(None)),
                j_slice + j_update,
            )
        else:
            Jmatrix[arrays[1], :] + j_update

        df_duT.append(time_derivs)

    return df_duT


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
    block_ind,
    blocks,
    field_derivs,
    fields,
    time_mask,
    Jmatrix,
):
    """
    Compute the rows of the sensitivity matrix for a given source and receiver.
    """
    rows = []
    for ind, (address, ind_array) in enumerate(blocks[block_ind]):
        # for (address, ind_array), field_derivs in zip(chunks, ATinv_df_duT_v):
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
            field_derivs[block_ind][ind][:, local_ind],
            adjoint=True,
        )

        dRHST_dm_v = simulation.getRHSDeriv(
            tInd + 1, src, field_derivs[block_ind][ind][:, local_ind], adjoint=True
        )  # on nodes of time mesh

        un_src = fields[:, address[0], tInd + 1]
        # cell centered on time mesh
        dAT_dm_v = simulation.getAdiagDeriv(
            tInd, un_src, field_derivs[block_ind][ind][:, local_ind], adjoint=True
        )
        row_block = np.zeros(
            (len(ind_array[1]), simulation.model.size), dtype=np.float32
        )
        row_block[time_check, :] = (-dAT_dm_v - dAsubdiagT_dm_v + dRHST_dm_v).T.astype(
            np.float32
        )

        if isinstance(Jmatrix, zarr.Array):
            j_slice = Jmatrix.get_orthogonal_selection((ind_array[1], slice(None)))
            Jmatrix.set_orthogonal_selection(
                (ind_array[1], slice(None)), j_slice + row_block
            )
        else:
            Jmatrix[ind_array[1], :] += row_block


def evaluate_dpred_block(indices, sources, mesh, time_mesh, fields):
    """
    Evaluate the data prediction for a block of sources.
    """
    data = []
    for ind in indices:

        receiver_list = sources[ind].receiver_list
        if len(receiver_list) == 0:
            continue

        for receiver in receiver_list:
            data.append(receiver.eval(sources[ind], mesh, time_mesh, fields))

    return np.hstack(data)


def dpred(self, m=None, f=None):
    # Docstring inherited from BaseSimulation.
    if self.survey is None:
        raise AttributeError(
            "The survey has not yet been set and is required to compute "
            "data. Please set the survey for the simulation: "
            "simulation.survey = survey"
        )

    client, worker = self._get_client_worker()

    if f is None:
        f = self.fields(m)

    delayed_chunks = []

    source_block = np.array_split(
        np.arange(len(self.survey.source_list)),
        self.n_threads(client=client, worker=worker),
    )
    if client:
        mesh = client.scatter(self.mesh, workers=worker)
        time_mesh = client.scatter(self.time_mesh, workers=worker)
        fields = client.scatter(f, workers=worker)
        source_list = client.scatter(self.survey.source_list, workers=worker)
    else:
        mesh = self.mesh
        time_mesh = self.time_mesh
        delayed_eval = delayed(evaluate_dpred_block)
        source_list = self.survey.source_list
        fields = f

    for block in source_block:
        if len(block) == 0:
            continue

        if client:
            delayed_chunks.append(
                client.submit(
                    evaluate_dpred_block,
                    block,
                    source_list,
                    mesh,
                    time_mesh,
                    fields,
                    workers=worker,
                )
            )
        else:
            delayed_chunks.append(
                delayed_eval(block, source_list, mesh, time_mesh, fields)
            )

    if client:
        result = client.gather(delayed_chunks)
    else:
        result = compute(delayed_chunks)[0]

    return np.hstack(result)


Sim.dpred = dpred
Sim.fields = fields
Sim.getSourceTerm = getSourceTerm
Sim.compute_J = compute_J
Sim.getJtJdiag = getJtJdiag
Sim.Jvec = Jvec
Sim.Jtvec = Jtvec
Sim.Jmatrix = Jmatrix
