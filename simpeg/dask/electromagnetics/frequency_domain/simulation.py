import gc

from ....electromagnetics.frequency_domain.simulation import BaseFDEMSimulation as Sim
from ....utils import Zero
from ...simulation import getJtJdiag, Jvec, Jtvec, Jmatrix
import numpy as np
import scipy.sparse as sp

# from multiprocessing import cpu_count
from dask import array, compute, delayed
from simpeg.dask.utils import get_parallel_blocks
from simpeg.electromagnetics.natural_source.sources import PlanewaveXYPrimary
import zarr


def evaluate_receivers(block, mesh, fields):
    data = []
    for source, _, receiver in block:
        data.append(receiver.eval(source, mesh, fields).flatten())

    return np.hstack(data)


def source_evaluation(simulation, sources):
    s_m, s_e = [], []
    for source in sources:
        sm, se = source.eval(simulation)
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


def eval_block(simulation, Ainv_deriv_u, deriv_indices, deriv_m, fields, address):
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

    return np.array(du_dmT, dtype=complex).reshape((du_dmT.shape[0], -1)).real.T


def getSourceTerm(self, freq, source=None):
    """
    Assemble the source term. This ensures that the RHS is a vector / array
    of the correct size
    """
    if source is None:
        # if self.client:
        #     n_splits = int(self.client.cluster.scheduler.total_nthreads / len(self.client.cluster.scheduler.workers))
        # else:
        #     n_splits = cpu_count()
        #
        # source_list = self.survey.get_sources_by_frequency(freq)
        # source_block = np.array_split(source_list, n_splits)
        #
        # block_compute = []
        #
        # if self.client:
        #     sim = self.client.scatter(self)
        #     source_block = self.client.scatter(source_block)
        #
        # for block in source_block:
        #     if self.client:
        #         block_compute.append(self.client.submit(source_evaluation, sim, block))
        #     else:
        #         block_compute.append(delayed(source_evaluation)(self, block))
        #
        # if self.client:
        #     blocks = self.client.gather(block_compute)
        # else:
        #     blocks = compute(block_compute)[0]
        s_m, s_e = [], []
        # for block in blocks:
        #     if block[0]:
        for source in self.survey.get_sources_by_frequency(freq):
            sm, se = source.eval(self)
            s_m.append(sm)
            s_e.append(se)
            # s_m += block[0]
            # s_e += block[1]

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


def dpred(self, m=None, f=None):
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
        f = self.fields(m)

    if self.client:
        f = self.client.scatter(f)
        mesh = self.client.scatter(self.mesh)
    else:
        mesh = delayed(self.mesh)
        delayed_block_eval = delayed(evaluate_receivers)

    rows = []
    for ind, src in enumerate(self.survey.source_list):
        for rx in src.receiver_list:
            block = [(src, ind, rx)]

            # receiver_blocks = np.array_split(np.asarray(all_receivers), cpu_count())
            # rows = []
            # for block in receiver_blocks:
            #         n_data = np.sum([rec.nD for _, _, rec in block])
            if rx.nD == 0:
                continue

            if self.client:
                rows.append(self.client.submit(evaluate_receivers, block, mesh, f))
            else:
                rows.append(
                    array.from_delayed(
                        delayed_block_eval(block, mesh, f),
                        dtype=np.float64,
                        shape=(rx.nD,),
                    )
                )

    if self.client:
        data = np.hstack(self.client.gather(rows))
    else:
        data = compute(array.hstack(rows))[0]

    return data


def fields(self, m=None):
    if m is not None:
        self.model = m

    if getattr(self, "_stashed_fields", None) is not None:
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

    self.Ainv = Ainv

    self._stashed_fields = f

    return f


def compute_J(self, m, f=None):
    self.model = m

    if f is None:
        f = self.fields(m)

    if len(self.Ainv) > 1:
        raise NotImplementedError(
            "Current implementation of parallelization assumes a single frequency per simulation. "
            "Consider creating one misfit per frequency."
        )

    A_i = list(self.Ainv.values())[0]
    m_size = m.size

    if self.store_sensitivities == "disk":
        Jmatrix = zarr.open(
            self.sensitivity_path,
            mode="w",
            shape=(self.survey.nD, m_size),
            chunks=(self.max_chunk_size, m_size),
        )
    else:
        Jmatrix = np.zeros((self.survey.nD, m_size), dtype=np.float32)

    compute_row_size = np.ceil(self.max_chunk_size / (A_i.A.shape[0] * 32.0 * 1e-6))
    blocks = get_parallel_blocks(
        self.survey.source_list, compute_row_size, optimize=False
    )
    fields_array = f[:, self._solutionType]
    blocks_receiver_derivs = []
    if self.client:
        for block in blocks:
            blocks_receiver_derivs.append(
                receiver_derivs(self.survey, self.mesh, f, block)
            )
    else:
        fields_array = delayed(f[:, self._solutionType])
        fields = delayed(f)
        survey = delayed(self.survey)
        mesh = delayed(self.mesh)

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
    if not self.client:
        blocks_receiver_derivs = compute(blocks_receiver_derivs)[0]

    for block_derivs_chunks, addresses_chunks in zip(blocks_receiver_derivs, blocks):
        Jmatrix = self.parallel_block_compute(
            m, Jmatrix, block_derivs_chunks, A_i, fields_array, addresses_chunks
        )

    for A in self.Ainv.values():
        A.clean()

    del self.Ainv
    gc.collect()
    if self.store_sensitivities == "disk":
        del Jmatrix
        Jmatrix = array.from_zarr(self.sensitivity_path)

    return Jmatrix


def parallel_block_compute(
    self, m, Jmatrix, blocks_receiver_derivs, A_i, fields_array, addresses
):
    m_size = m.size
    block_stack = sp.hstack(blocks_receiver_derivs).toarray()

    ATinvdf_duT = A_i * block_stack
    if not self.client:
        ATinvdf_duT = delayed(ATinvdf_duT)
    count = 0
    rows = []
    block_delayed = []

    for address, dfduT in zip(addresses, blocks_receiver_derivs):
        n_cols = dfduT.shape[1]
        n_rows = address[1][2]

        if self.client:
            block_delayed.append(
                eval_block(
                    self,
                    ATinvdf_duT,
                    np.arange(count, count + n_cols),
                    Zero(),
                    fields_array,
                    address,
                )
            )
        else:
            delayed_eval = delayed(eval_block)
            block_delayed.append(
                array.from_delayed(
                    delayed_eval(
                        self,
                        ATinvdf_duT,
                        np.arange(count, count + n_cols),
                        Zero(),
                        fields_array,
                        address,
                    ),
                    dtype=np.float32,
                    shape=(n_rows, m_size),
                )
            )
        count += n_cols
        rows += address[1][1].tolist()

    indices = np.hstack(rows)

    if self.client:
        block = np.vstack(block_delayed)
    else:
        block = compute(array.vstack(block_delayed))[0]

    if self.store_sensitivities == "disk":
        Jmatrix.set_orthogonal_selection(
            (indices, slice(None)),
            block,
        )
    else:
        # Dask process to compute row and store
        Jmatrix[indices, :] = block

    return Jmatrix


Sim.parallel_block_compute = parallel_block_compute
Sim.compute_J = compute_J
Sim.getJtJdiag = getJtJdiag
Sim.Jvec = Jvec
Sim.Jtvec = Jtvec
Sim.Jmatrix = Jmatrix
Sim.fields = fields
# Sim.dpred = dpred
Sim.getSourceTerm = getSourceTerm
