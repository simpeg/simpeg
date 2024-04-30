from ....electromagnetics.frequency_domain.simulation import BaseFDEMSimulation as Sim
from ....utils import Zero
import numpy as np
import scipy.sparse as sp
from multiprocessing import cpu_count
from dask import array, compute, delayed
from SimPEG.dask.simulation import dask_Jvec, dask_Jtvec, dask_getJtJdiag
from SimPEG.dask.utils import get_parallel_blocks
from SimPEG.electromagnetics.natural_source.sources import PlanewaveXYPrimary
import zarr
from time import time

Sim.sensitivity_path = "./sensitivity/"
Sim.gtgdiag = None
Sim.store_sensitivities = True

Sim.getJtJdiag = dask_getJtJdiag
Sim.Jvec = dask_Jvec
Sim.Jtvec = dask_Jtvec
Sim.clean_on_model_update = ["_Jmatrix", "_jtjdiag"]


@delayed
def evaluate_receivers(block, mesh, fields):
    data = []
    for source, ind, receiver in block:
        data.append(receiver.eval(source, mesh, fields).flatten())

    return np.hstack(data)


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

    all_receivers = []

    for ind, src in enumerate(self.survey.source_list):
        for rx in src.receiver_list:
            all_receivers.append((src, ind, rx))

    receiver_blocks = np.array_split(all_receivers, cpu_count())
    rows = []
    for block in receiver_blocks:
        n_data = np.sum(rec.nD for _, _, rec in block)
        if n_data == 0:
            continue

        rows.append(
            array.from_delayed(
                evaluate_receivers(block, self.mesh, f),
                dtype=np.float64,
                shape=(n_data,),
            )
        )

    data = array.hstack(rows).compute()

    if compute_J and self._Jmatrix is None:
        Jmatrix = self.compute_J(f=f, Ainv=Ainv)
        return data, Jmatrix

    return data


Sim.dpred = dask_dpred
Sim.field_derivs = None


def fields(self, m=None, return_Ainv=False):
    if m is not None:
        self.model = m

    f = self.fieldsPair(self)

    Ainv = {}
    for freq in self.survey.frequencies:
        A = self.getA(freq)
        rhs = self.getRHS(freq)
        Ainv_solve = self.solver(sp.csr_matrix(A), **self.solver_opts)
        u = Ainv_solve * rhs
        sources = self.survey.get_sources_by_frequency(freq)
        f[sources, self._solutionType] = u

        if return_Ainv:
            Ainv[freq] = Ainv_solve
        else:
            Ainv_solve.clean()

    if return_Ainv:
        return f, Ainv
    else:
        return f, None


Sim.fields = fields


def compute_J(self, f=None, Ainv=None):
    if f is None:
        f, Ainv = self.fields(self.model, return_Ainv=True)

    if len(Ainv) > 1:
        raise NotImplementedError(
            "Current implementation of parallelization assumes a single frequency per simulation. "
            "Consider creating one misfit per frequency."
        )

    A_i = list(Ainv.values())[0]
    m_size = self.model.size

    if self.store_sensitivities == "disk":
        Jmatrix = zarr.open(
            self.sensitivity_path,
            mode="w",
            shape=(self.survey.nD, m_size),
            chunks=(row_chunks, m_size),
        )
    else:
        Jmatrix = np.zeros((self.survey.nD, m_size), dtype=np.float32)

    compute_row_size = np.ceil(self.max_chunk_size / (A_i.A.shape[0] * 32.0 * 1e-6))
    blocks = get_parallel_blocks(
        self.survey.source_list, compute_row_size, optimize=False
    )
    count = 0
    fields_array = delayed(f[:, self._solutionType])

    for block in blocks:
        addresses = []
        blocks_receiver_derivs = []
        chunks = np.array_split(np.arange(len(block)), cpu_count())

        for chunk in chunks:
            if len(chunk) == 0:
                continue

            n_fields = np.sum(
                [len(elem[1][0]) for elem in block[chunk[0] : chunk[0] + len(chunk)]]
            )

            shape = [A_i.A.shape[0], n_fields]

            if isinstance(self.survey.source_list[0], PlanewaveXYPrimary):
                shape[1] *= 2

            blocks_receiver_derivs.append(
                array.from_delayed(
                    receiver_derivs(
                        self.survey,
                        self.mesh,
                        f,
                        block[chunk[0] : chunk[0] + len(chunk)],
                    ),
                    dtype=np.complex128,
                    shape=shape,
                )
            )
            addresses.append(block[chunk[0] : chunk[0] + len(chunk)])

        Jmatrix = parallel_block_compute(
            self, Jmatrix, blocks_receiver_derivs, A_i, fields_array, addresses
        )
    #     addresses = []
    #     blocks_receiver_derivs = []
    #     count = 0
    #
    # if blocks_receiver_derivs:
    #     Jmatrix = parallel_block_compute(
    #         self,
    #         Jmatrix,
    #         blocks_receiver_derivs,
    #         Ainv[src.frequency],
    #         fields_array,
    #         addresses,
    #     )

    for A in Ainv.values():
        A.clean()

    if self.store_sensitivities == "disk":
        del Jmatrix
        return array.from_zarr(self.sensitivity_path + f"J.zarr")
    else:
        return Jmatrix


Sim.compute_J = compute_J


def parallel_block_compute(
    self, Jmatrix, blocks_receiver_derivs, A_i, fields_array, addresses
):
    print("In parallel block")
    m_size = self.model.size

    # tc = time()
    # print(f"Compute blocks_receiver_derivs {len(blocks_receiver_derivs)}")
    # eval = compute(blocks_receiver_derivs)[0]
    # print(f"Compute blocks_receiver_derivs time: {time() - tc}")
    # blocks_dfduT, blocks_dfdmT = [], []
    # for dfduT, dfdmT in eval:
    #     blocks_dfduT.append(dfduT)
    #     blocks_dfdmT.append(dfdmT)

    tc = time()
    print(f"Compute block stack {len(blocks_receiver_derivs)}")
    block_stack = array.hstack(blocks_receiver_derivs).compute()
    print(f"Compute block stack time: {time() - tc}")

    tc = time()
    print(f"Compute direct solver")
    ATinvdf_duT = delayed(A_i * block_stack)
    print(f"Compute direct solver time: {time() - tc}")
    count = 0
    rows = []
    block_delayed = []
    tc = time()
    print("Loop over addresses")
    for block_addresses, dfduT in zip(addresses, blocks_receiver_derivs):
        n_cols = dfduT.shape[1]
        n_rows = np.sum([address[1][2] for address in block_addresses])
        block_delayed.append(
            array.from_delayed(
                eval_block(
                    self,
                    ATinvdf_duT,
                    np.arange(count, count + n_cols),
                    Zero(),
                    fields_array,
                    block_addresses
                    # src,
                    # address[0][0],
                ),
                dtype=np.float32,
                shape=(n_rows, m_size),
            )
        )
        count += n_cols
        rows += [address[1][1] for address in block_addresses]

    print(f"Loop over addresses time: {time() - tc}")

    indices = np.hstack(rows)

    if self.store_sensitivities == "disk":
        Jmatrix.set_orthogonal_selection(
            (np.r_[rows], slice(None)),
            array.vstack(block_delayed).compute(),
        )
    else:
        tc = time()
        print("Compute Jmatrix")
        Jmatrix[indices, :] = array.vstack(block_delayed).compute()
        print(f"Compute Jmatrix time: {time() - tc}")

    return Jmatrix


@delayed
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

    field_derivatives = sp.hstack(field_derivatives)
    return field_derivatives.toarray()


@delayed
def eval_block(simulation, Ainv_deriv_u, deriv_indices, deriv_m, fields, addresses):
    """
    Evaluate the sensitivities for the block or data
    """
    count = 0
    rows = []
    if Ainv_deriv_u.ndim == 1:
        deriv_columns = Ainv_deriv_u[:, np.newaxis]
    else:
        deriv_columns = Ainv_deriv_u[:, deriv_indices]

    for address in addresses:
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
            deriv_columns[:, count : count + n_cols],
            adjoint=True,
        )
        dRHS_dmT = simulation.getRHSDeriv(
            source.frequency,
            source,
            deriv_columns[:, count : count + n_cols],
            adjoint=True,
        )
        du_dmT = -dA_dmT
        if not isinstance(dRHS_dmT, Zero):
            du_dmT += dRHS_dmT
        if not isinstance(deriv_m, Zero):
            du_dmT += deriv_m

        rows.append(
            np.array(du_dmT, dtype=complex).reshape((du_dmT.shape[0], -1)).real.T
        )
        count += n_cols

    return np.vstack(rows)
