import dask
import dask.array

from ....electromagnetics.time_domain.simulation import BaseTDEMSimulation as Sim
from ....utils import Zero
import numpy as np
import scipy.sparse as sp
from time import time
from dask import array, compute, delayed
from SimPEG.dask.simulation import dask_Jvec, dask_Jtvec, dask_getJtJdiag
import zarr
from SimPEG.utils import mkvc
from tqdm import tqdm
Sim.sensitivity_path = './sensitivity/'
Sim.gtgdiag = None
Sim.store_sensitivities = True

Sim.getJtJdiag = dask_getJtJdiag
Sim.Jvec = dask_Jvec
Sim.Jtvec = dask_Jtvec
Sim.clean_on_model_update = ["_Jmatrix", "gtgdiag"]


def fields(self, m=None, return_Ainv=False):
    if m is not None:
        self.model = m

    f = self.fieldsPair(self)

    # set initial fields
    f[:, self._fieldType + "Solution", 0] = self.getInitialFields()

    Ainv = {}
    ATinv = {}
    for tInd, dt in enumerate(self.time_steps):

        if dt not in Ainv:
            A = self.getAdiag(tInd)
            Ainv[dt] = self.solver(sp.csr_matrix(A), **self.solver_opts)
            if return_Ainv:
                ATinv[dt] = self.solver(sp.csr_matrix(A.T), **self.solver_opts)

        rhs = self.getRHS(tInd + 1)
        Asubdiag = self.getAsubdiag(tInd)
        sol = Ainv[dt] * (rhs - Asubdiag * f[:, (self._fieldType + "Solution"), tInd])
        f[:, self._fieldType + "Solution", tInd + 1] = sol

    for A in Ainv.values():
        A.clean()

    if return_Ainv:
        return f, ATinv
    else:
        return f, None


Sim.fields = fields


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

    def evaluate_receiver(source, receiver, mesh, time_mesh, fields):
        return receiver.eval(source, mesh, time_mesh, fields).flatten()

    row = delayed(evaluate_receiver, pure=True)
    rows = []
    for src in self.survey.source_list:
        for rx in src.receiver_list:
            rows.append(array.from_delayed(
                row(src, rx, self.mesh, self.time_mesh, f),
                dtype=np.float32,
                shape=(rx.nD,),
            ))

    data = array.hstack(rows).compute()

    if compute_J and self._Jmatrix is None:
        Jmatrix = self.compute_J(f=f, Ainv=Ainv)
        return data, Jmatrix

    return data


Sim.dpred = dask_dpred
Sim.field_derivs = None

def compute_J(self, f=None, Ainv=None):

    if f is None:
        f, Ainv = self.fields(self.model, return_Ainv=True)

    m_size = self.model.size
    row_chunks = int(np.ceil(
        float(self.survey.nD) / np.ceil(float(m_size) * self.survey.nD * 8. * 1e-6 / self.max_chunk_size)
    ))

    if self.store_sensitivities == "disk":
        self.J_initializer = zarr.open(
            self.sensitivity_path + f"J_initializer.zarr",
            mode='w',
            shape=(self.survey.nD, m_size),
            chunks=(row_chunks, m_size)
        )
    else:
        self.J_initializer = np.zeros((self.survey.nD, m_size), dtype=np.float32)
    solution_type = self._fieldType + "Solution"  # the thing we solved for

    if self.field_derivs is None:

        # print("Start loop for field derivs")
        block_size = len(f[self.survey.source_list[0], solution_type, 0])

        field_derivs = []
        for tInd in range(self.nT + 1):
            d_count = 0
            df_duT_v = []
            for i_s, src in enumerate(self.survey.source_list):
                src_field_derivs = delayed(block_deriv, pure=True)(self, src, tInd, f, block_size, d_count)
                df_duT_v += [src_field_derivs]
                d_count += np.sum([rx.nD for rx in src.receiver_list])

            field_derivs += [df_duT_v]
        # print("Dask loop field derivs")
        # tc = time()

        self.field_derivs = dask.compute(field_derivs)[0]
        # print(f"Done in {time() - tc} seconds")

    if self.store_sensitivities == "disk":
        Jmatrix = zarr.open(
            self.sensitivity_path + f"J.zarr",
            mode='w',
            shape=(self.survey.nD, m_size),
            chunks=(row_chunks, m_size)
        ) + self.J_initializer
    else:
        Jmatrix = dask.delayed(np.zeros((self.survey.nD, m_size), dtype=np.float32) + self.J_initializer)

    # ATinv_df_duT_v = {}
    f = dask.delayed(f)
    field_derivs_t = {}
    for tInd, dt in tqdm(zip(reversed(range(self.nT)), reversed(self.time_steps))):
        AdiagTinv = Ainv[dt]
        Asubdiag = self.getAsubdiag(tInd)
        d_count = 0
        row_blocks = []

        # tc_loop = time()
        # print(f"Loop sources for {tInd}")
        for isrc, src in enumerate(self.survey.source_list):
            source_blocks = []
            # for block in range(len(self.field_derivs[tInd][isrc])):
            if isrc not in field_derivs_t:
                ATinv_df_duT_v = dask.delayed(AdiagTinv * self.field_derivs[tInd+1][isrc].toarray())
            else:
                ATinv_df_duT_v = dask.delayed(AdiagTinv * field_derivs_t[isrc])

            n_data = self.field_derivs[tInd+1][isrc].shape[1]
            n_blocks = int(np.ceil((m_size * n_data) * 8. * 1e-6 / 128.))
            ind_col = np.array_split(np.arange(n_data), n_blocks)

            for col_block in ind_col:
                source_blocks.append(
                    dask.array.from_delayed(
                        delayed(parallel_block_compute, pure=True)(
                            self, f, src, ATinv_df_duT_v, d_count,
                            col_block, tInd, solution_type, Jmatrix, Asubdiag,
                            self.field_derivs[tInd][isrc]
                        ),
                        shape=self.field_derivs[tInd+1][isrc].shape,
                        dtype=np.float32
                    )
                )
                # print(f"Appending block {isrc} in {time() - tc} seconds")
                d_count += len(col_block)

            row_blocks.append(dask.array.hstack(source_blocks))
        # print(f"Done in {time() - tc_loop} seconds")
        # tc = time()
        # print(f"Compute field derivs for {tInd}")
        del field_derivs_t
        field_derivs_t = {isrc: elem for isrc, elem in enumerate(dask.compute(row_blocks)[0])}
        # print(f"Done in {time() - tc} seconds")

    for A in Ainv.values():
        A.clean()

    if self.store_sensitivities == "disk":
        del Jmatrix
        return array.from_zarr(self.sensitivity_path + f"J.zarr")
    else:
        return Jmatrix.compute()

Sim.compute_J = compute_J


def block_deriv(simulation, src, tInd, f, block_size, d_count):
    src_field_derivs = None
    for rx in src.receiver_list:

        v = sp.eye(rx.nD, dtype=float)
        PT_v = rx.evalDeriv(
            src, simulation.mesh, simulation.time_mesh, f, v, adjoint=True
        )
        df_duTFun = getattr(f, "_{}Deriv".format(rx.projField), None)

        cur = df_duTFun(
            simulation.nT,
            src,
            None,
            PT_v[tInd * block_size:(tInd + 1) * block_size, :],
            adjoint=True,
        )

        if not isinstance(cur[1], Zero):
            simulation.J_initializer[d_count:d_count + rx.nD, :] += cur[1].T

        if src_field_derivs is None:
            src_field_derivs = cur[0]
        else:
            src_field_derivs += cur[0]

    # n_blocks = int(np.ceil(np.prod(src_field_derivs.shape) * 8. * 1e-6 / 128.))
    # ind_col = np.array_split(np.arange(src_field_derivs.shape[1]), col_blocks)
    # return [src_field_derivs[:, ind] for ind in ind_col]
    return src_field_derivs

def parallel_block_compute(simulation, f, src, ATinv_df_duT_v, d_count, col_block, tInd, solution_type, Jmatrix, Asubdiag, field_derivs):
    field_derivs_t = (
        field_derivs[:, col_block]
        - Asubdiag.T * ATinv_df_duT_v[:, col_block]
    )

    dAsubdiagT_dm_v = simulation.getAsubdiagDeriv(
        tInd, f[src, solution_type, tInd], ATinv_df_duT_v[:, col_block], adjoint=True
    )

    dRHST_dm_v = simulation.getRHSDeriv(
        tInd + 1, src, ATinv_df_duT_v[:, col_block], adjoint=True
    )
    un_src = f[src, solution_type, tInd + 1]
    dAT_dm_v = simulation.getAdiagDeriv(
        tInd, un_src, ATinv_df_duT_v[:, col_block], adjoint=True
    )
    Jmatrix[d_count:d_count + dAT_dm_v.shape[1], :] += (-dAT_dm_v - dAsubdiagT_dm_v + dRHST_dm_v).T

    return field_derivs_t