import gc

from ....electromagnetics.frequency_domain.simulation import BaseFDEMSimulation as Sim
from ....utils import Zero
from ...simulation import getJtJdiag, Jvec, Jtvec, Jmatrix
import numpy as np
import scipy.sparse as sp

from dask import array, compute, delayed
from simpeg.dask.utils import get_parallel_blocks
from simpeg.electromagnetics.natural_source.sources import PlanewaveXYPrimary
import zarr
from dask.distributed import get_client


def receivers_eval(block, mesh, fields):
    data = []
    for source, _, receiver in block:
        data.append(receiver.eval(source, mesh, fields).flatten())

    return np.hstack(data)


def source_eval(simulation, sources):
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

        source_list = self.survey.get_sources_by_frequency(freq)
        block_compute = []

        for block in [source_list]:
            block_compute.append(source_eval(self, block))

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


def dpred(self, m=None, f=None, compute_J=False):
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

    all_receivers = []

    for ind, src in enumerate(self.survey.source_list):
        for rx in src.receiver_list:
            all_receivers.append((src, ind, rx))

    receiver_blocks = np.array_split(np.asarray(all_receivers), self.n_threads)
    rows = []

    if self.client:
        f = self.client.scatter(f, workers=self.worker)
        mesh = self.client.scatter(self.mesh, workers=self.worker)
    else:
        delayed_receivers_eval = delayed(receivers_eval)
        mesh = delayed(self.mesh)

    for block in receiver_blocks:
        n_data = np.sum([rec.nD for _, _, rec in block])
        if n_data == 0:
            continue

        if self.client:
            rows.append(
                self.client.submit(receivers_eval, block, mesh, f, workers=self.worker)
            )
        else:
            rows.append(
                array.from_delayed(
                    delayed_receivers_eval(block, mesh, f),
                    dtype=np.float64,
                    shape=(n_data,),
                )
            )

    if self.client:
        rows = np.hstack(self.client.gather(rows))
    else:
        rows = compute(array.hstack(rows))[0]

    return rows


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

    # self.Ainv = Ainv

    self._stashed_fields = f

    if return_Ainv:
        return f, Ainv
    return f


def compute_J(self, m, f=None):
    self.model = m

    client = get_client()

    if f is None:
        f, Ainv = self.fields(m=m, return_Ainv=True)

    if len(Ainv) > 1:
        raise NotImplementedError(
            "Current implementation of parallelization assumes a single frequency per simulation. "
            "Consider creating one misfit per frequency."
        )

    A_i = list(Ainv.values())[0]
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
    if client:
        fields_array = client.scatter(f[:, self._solutionType], workers=self.worker)
        fields = client.scatter(f, workers=self.worker)
        survey = client.scatter(self.survey, workers=self.worker)
        mesh = client.scatter(self.mesh, workers=self.worker)
        for block in blocks:
            blocks_receiver_derivs.append(
                client.submit(
                    receiver_derivs,
                    survey,
                    mesh,
                    fields,
                    block,
                    workers=self.worker,
                )
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
    if client:
        blocks_receiver_derivs = client.gather(blocks_receiver_derivs)
    else:
        blocks_receiver_derivs = compute(blocks_receiver_derivs)[0]

    for block_derivs_chunks, addresses_chunks in zip(blocks_receiver_derivs, blocks):
        Jmatrix = self.parallel_block_compute(
            m, Jmatrix, block_derivs_chunks, A_i, fields_array, addresses_chunks, client
        )

    for A in Ainv.values():
        A.clean()

    del Ainv
    gc.collect()
    if self.store_sensitivities == "disk":
        del Jmatrix
        Jmatrix = array.from_zarr(self.sensitivity_path)

    return Jmatrix


def parallel_block_compute(
    self, m, Jmatrix, blocks_receiver_derivs, A_i, fields_array, addresses, client
):
    m_size = m.size
    block_stack = sp.hstack(blocks_receiver_derivs).toarray()

    ATinvdf_duT = A_i * block_stack
    if client:
        ATinvdf_duT = client.scatter(ATinvdf_duT, workers=self.worker)
    else:
        ATinvdf_duT = delayed(ATinvdf_duT)
    count = 0
    rows = []
    block_delayed = []

    for address, dfduT in zip(addresses, blocks_receiver_derivs):
        n_cols = dfduT.shape[1]
        n_rows = address[1][2]

        if client:
            sim = client.scatter(self, workers=self.worker)
            block_delayed.append(
                client.submit(
                    eval_block,
                    sim,
                    ATinvdf_duT,
                    np.arange(count, count + n_cols),
                    Zero(),
                    fields_array,
                    address,
                    workers=self.worker,
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

    if client:
        block_delayed = client.gather(block_delayed)
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
