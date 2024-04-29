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

    compute_row_size = np.ceil(self.max_chunk_size / (A_i.A.shape[0] * 16.0 * 1e-6))
    blocks = get_parallel_blocks(self.survey.source_list, compute_row_size)
    count = 0
    fields_array = f[:, self._solutionType]
    addresses = []
    blocks_receiver_derivs = []

    for block in blocks:
        chunk_ind = [0]

        for address in block:
            src = self.survey.source_list[address[0][0]]
            rx = src.receiver_list[address[0][1]]
            # v = sp.diags(np.ones(rx.nD), dtype=float, format="csr")[:, address[1][0]]
            v = np.eye(rx.nD, dtype=float)[:, address[1][0]]

            blocks_receiver_derivs.append(receiver_derivs(src, rx, self.mesh, f, v))

            count += len(address[1][0])
            addresses.append(address)

            if count > compute_row_size * cpu_count():
                Jmatrix = parallel_block_compute(
                    self, Jmatrix, blocks_receiver_derivs, A_i, fields_array, addresses
                )
                addresses = []
                blocks_receiver_derivs = []
                count = 0

    if blocks_receiver_derivs:
        Jmatrix = parallel_block_compute(
            self,
            Jmatrix,
            blocks_receiver_derivs,
            Ainv[src.frequency],
            fields_array,
            addresses,
        )

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

    tc = time()
    print(f"Compute blocks_receiver_derivs")
    eval = compute(blocks_receiver_derivs)[0]
    print(f"Compute blocks_receiver_derivs time: {time() - tc}")
    blocks_dfduT, blocks_dfdmT = [], []
    for dfduT, dfdmT in eval:
        blocks_dfduT.append(dfduT)
        blocks_dfdmT.append(dfdmT)

    tc = time()
    print(f"Compute direct solver")
    ATinvdf_duT = (A_i * array.hstack(blocks_dfduT).compute()).reshape(
        (fields_array.shape[0], -1)
    )
    print(f"Compute direct solver time: {time() - tc}")
    count = 0
    rows = []
    block_delayed = []
    tc = time()
    print("Loop over addresses")
    for address, dfdmT, dfduT in zip(addresses, blocks_dfdmT, blocks_dfduT):
        n_cols = dfduT.shape[1]
        src = self.survey.source_list[address[0][0]]
        if isinstance(src, PlanewaveXYPrimary):
            u_src = fields_array
        else:
            u_src = fields_array[:, address[0][0]]

        block_delayed.append(
            array.from_delayed(
                delayed(eval_block, pure=True)(
                    self, ATinvdf_duT[:, count : count + n_cols], dfdmT, u_src, src
                ),
                dtype=np.float32,
                shape=(len(address[1][1]), m_size),
            )
        )
        count += n_cols
        rows.append(address[1][1])

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
def receiver_derivs(source, receiver, mesh, fields, block):
    dfduT, dfdmT = receiver.evalDeriv(source, mesh, fields, v=block, adjoint=True)

    return dfduT, dfdmT


def eval_block(simulation, Ainv_deriv_u, deriv_m, fields, source):
    """
    Evaluate the sensitivities for the block or data and store to zarr
    """
    dA_dmT = simulation.getADeriv(source.frequency, fields, Ainv_deriv_u, adjoint=True)
    dRHS_dmT = simulation.getRHSDeriv(
        source.frequency, source, Ainv_deriv_u, adjoint=True
    )
    du_dmT = -dA_dmT
    if not isinstance(dRHS_dmT, Zero):
        du_dmT += dRHS_dmT
    if not isinstance(deriv_m, Zero):
        du_dmT += deriv_m

    return np.array(du_dmT, dtype=complex).reshape((du_dmT.shape[0], -1)).real.T
