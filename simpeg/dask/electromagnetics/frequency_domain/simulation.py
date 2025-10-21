import gc
import os
import shutil

from ....electromagnetics.frequency_domain.simulation import BaseFDEMSimulation as Sim
from ....utils import Zero
from ...simulation import getJtJdiag, Jvec, Jtvec, Jmatrix
import numpy as np
import scipy.sparse as sp

from dask import array, compute, delayed
from simpeg.dask.utils import get_parallel_blocks
from simpeg.electromagnetics.natural_source.sources import PlanewaveXYPrimary
import zarr


def receivers_eval(block, mesh, fields):
    data = []
    for source, _, receiver in block:
        data.append(receiver.eval(source, mesh, fields).flatten())

    return np.hstack(data)


def source_eval(simulation, sources, indices):
    s_m, s_e = [], []
    for ind in indices:
        sm, se = sources[ind].eval(simulation)
        s_m.append(sm)
        s_e.append(se)

    return s_m, s_e


def receiver_derivs(survey, mesh, fields, blocks):
    field_derivatives = []
    for address in blocks:
        source = survey.source_list[address[0][0]]
        receiver = source.receiver_list[address[0][1]]

        if isinstance(source, PlanewaveXYPrimary):
            v = np.eye(receiver.nD, dtype=float)
        else:
            v = sp.csr_matrix(np.ones(receiver.nD), dtype=float)

        # Assume the derivatives in terms of model are Zero (seems to always be case)
        dfduT, _ = receiver.evalDeriv(
            source, mesh, fields, v=v[:, address[1][0]], adjoint=True
        )
        field_derivatives.append(dfduT)

    return field_derivatives


def compute_rows(
    simulation, Ainv_deriv_u, deriv_indices, deriv_m, fields, address, Jmatrix
):
    """
    Evaluate the sensitivities for the block or data
    """

    if Ainv_deriv_u.ndim == 1:
        deriv_columns = Ainv_deriv_u[:, np.newaxis]
    else:
        deriv_columns = Ainv_deriv_u[:, deriv_indices]

    n_receivers = address[1][2]
    source = simulation.survey.source_list[address[0][0]]

    if isinstance(source, PlanewaveXYPrimary):
        source_fields = fields
        n_cols = 2
    else:
        source_fields = fields[:, address[0][0]]
        n_cols = 1

    n_cols *= n_receivers

    dA_dmT = simulation.getADeriv(
        source.frequency,
        source_fields,
        deriv_columns,
        adjoint=True,
    )

    dRHS_dmT = simulation.getRHSDeriv(
        source.frequency,
        source,
        deriv_columns,
        adjoint=True,
    )

    du_dmT = -dA_dmT
    if not isinstance(dRHS_dmT, Zero):
        du_dmT += dRHS_dmT
    if not isinstance(deriv_m, Zero):
        du_dmT += deriv_m

    values = np.array(du_dmT, dtype=complex).reshape((du_dmT.shape[0], -1)).real.T

    if isinstance(Jmatrix, zarr.Array):
        Jmatrix.set_orthogonal_selection((address[1][1], slice(None)), values)
    else:
        Jmatrix[address[1][1], :] = values

    return None


def getSourceTerm(self, freq, source=None):
    """
    Assemble the source term. This ensures that the RHS is a vector / array
    of the correct size
    """

    if source is None:

        client, worker = self._get_client_worker()
        source_list = self.survey.get_sources_by_frequency(freq)
        source_blocks = np.array_split(
            np.arange(len(source_list)), self.n_threads(client=client, worker=worker)
        )

        if client:
            sim = client.scatter(self, workers=self.worker)
            source_list = client.scatter(source_list, workers=worker)
        else:
            sim = self

        block_compute = []

        for block in source_blocks:
            if len(block) == 0:
                continue

            if client:
                block_compute.append(
                    client.submit(source_eval, sim, source_list, block, workers=worker)
                )
            else:
                block_compute.append(source_eval(sim, source_list, block))

        if client:
            block_compute = client.gather(block_compute)

        s_m, s_e = [], []
        for block in block_compute:
            if block[0]:
                s_m += block[0]
                s_e += block[1]

    else:
        sm, se = source.eval(self)
        s_m, s_e = [sm], [se]

    if isinstance(s_m[0][0], Zero):  # Assume the rest is all Zero
        s_m = Zero()
    else:
        s_m = np.vstack(s_m)
        if s_m.shape[0] < s_m.shape[1]:
            s_m = s_m.T

    if isinstance(s_e[0][0], Zero):  # Assume the rest is all Zero
        s_e = Zero()
    else:
        s_e = np.vstack(s_e)
        if s_e.shape[0] < s_e.shape[1]:
            s_e = s_e.T

    return s_m, s_e


def fields(self, m=None, return_Ainv=False):
    if m is not None:
        self.model = m

    if getattr(self, "_stashed_fields", None) is not None and not return_Ainv:
        return self._stashed_fields

    f = self.fieldsPair(self)
    Ainv = {}
    for freq in self.survey.frequencies:
        A = self.getA(freq)
        rhs = self.getRHS(freq)
        Ainv_solve = self.solver(sp.csr_matrix(A), **self.solver_opts)
        u = Ainv_solve * rhs
        sources = self.survey.get_sources_by_frequency(freq)
        f[sources, self._solutionType] = u
        Ainv[freq] = Ainv_solve

    self._stashed_fields = f

    if return_Ainv:
        return f, Ainv
    return f


def compute_J(self, m, f=None):
    self.model = m

    if f is None:
        f, Ainv = self.fields(m=m, return_Ainv=True)

    if len(Ainv) > 1:
        raise NotImplementedError(
            "Current implementation of parallelization assumes a single frequency per simulation. "
            "Consider creating one misfit per frequency."
        )

    client, worker = self._get_client_worker()

    A_i = list(Ainv.values())[0]
    m_size = m.size
    compute_row_size = np.ceil(self.max_chunk_size / (A_i.A.shape[0] * 32.0 * 1e-6))
    blocks = get_parallel_blocks(
        self.survey.source_list, compute_row_size, optimize=True
    )

    if self.store_sensitivities == "disk":

        chunk_size = np.median(
            [np.sum([len(chunk[1][1]) for chunk in block]) for block in blocks]
        ).astype(int)

        if os.path.exists(self.sensitivity_path):
            shutil.rmtree(self.sensitivity_path)

        Jmatrix = zarr.open(
            self.sensitivity_path,
            mode="w",
            shape=(self.survey.nD, m_size),
            chunks=(chunk_size, m_size),
        )
    else:
        Jmatrix = np.zeros((self.survey.nD, m_size), dtype=np.float32)

        if client:
            Jmatrix = client.scatter(Jmatrix, workers=worker)

    fields_array = f[:, self._solutionType]
    blocks_receiver_derivs = []

    if client:
        fields_array = client.scatter(f[:, self._solutionType], workers=worker)
        fields = client.scatter(f, workers=worker)
        survey = client.scatter(self.survey, workers=worker)
        mesh = client.scatter(self.mesh, workers=worker)
        simulation = client.scatter(self, workers=worker)
        for block in blocks:
            blocks_receiver_derivs.append(
                client.submit(
                    receiver_derivs,
                    survey,
                    mesh,
                    fields,
                    block,
                    workers=worker,
                )
            )
    else:
        fields_array = delayed(f[:, self._solutionType])
        fields = delayed(f)
        survey = delayed(self.survey)
        mesh = delayed(self.mesh)
        simulation = delayed(self)
        delayed_derivs = delayed(receiver_derivs)
        for block in blocks:
            blocks_receiver_derivs.append(
                delayed_derivs(
                    survey,
                    mesh,
                    fields,
                    block,
                )
            )

    # Dask process for all derivatives
    if client:
        blocks_receiver_derivs = client.gather(blocks_receiver_derivs)
    else:
        blocks_receiver_derivs = compute(blocks_receiver_derivs)[0]

    for block_derivs_chunks, addresses_chunks in zip(
        blocks_receiver_derivs, blocks, strict=True
    ):
        parallel_block_compute(
            simulation,
            m,
            Jmatrix,
            block_derivs_chunks,
            A_i,
            fields_array,
            addresses_chunks,
            client,
            worker,
        )

    for A in Ainv.values():
        A.clean()

    del Ainv
    gc.collect()
    if self.store_sensitivities == "disk":
        del Jmatrix
        return array.from_zarr(self.sensitivity_path)

    if client:
        return client.gather(Jmatrix)

    return Jmatrix


def parallel_block_compute(
    simulation,
    m,
    Jmatrix,
    blocks_receiver_derivs,
    A_i,
    fields_array,
    addresses,
    client,
    worker=None,
):
    m_size = m.size
    block_stack = sp.hstack(blocks_receiver_derivs).toarray()

    ATinvdf_duT = A_i * block_stack

    if client:
        ATinvdf_duT = client.scatter(ATinvdf_duT, workers=worker)
    else:
        ATinvdf_duT = delayed(ATinvdf_duT)

    count = 0
    block_delayed = []
    for address, dfduT in zip(addresses, blocks_receiver_derivs):
        n_cols = dfduT.shape[1]
        n_rows = address[1][2]

        if client:
            block_delayed.append(
                client.submit(
                    compute_rows,
                    simulation,
                    ATinvdf_duT,
                    np.arange(count, count + n_cols),
                    Zero(),
                    fields_array,
                    address,
                    Jmatrix,
                    workers=worker,
                )
            )
        else:
            delayed_eval = delayed(compute_rows)
            block_delayed.append(
                array.from_delayed(
                    delayed_eval(
                        simulation,
                        ATinvdf_duT,
                        np.arange(count, count + n_cols),
                        Zero(),
                        fields_array,
                        address,
                        Jmatrix,
                    ),
                    dtype=np.float32,
                    shape=(n_rows, m_size),
                )
            )
        count += n_cols

    if client:
        return client.gather(block_delayed)
    else:
        return compute(block_delayed)


Sim.compute_J = compute_J
Sim.getJtJdiag = getJtJdiag
Sim.Jvec = Jvec
Sim.Jtvec = Jtvec
Sim.Jmatrix = Jmatrix
Sim.fields = fields
Sim.getSourceTerm = getSourceTerm
