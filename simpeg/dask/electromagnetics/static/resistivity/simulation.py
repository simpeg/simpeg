from .....electromagnetics.static.resistivity.simulation import Simulation3DNodal as Sim

from ....simulation import getJtJdiag, Jvec, Jtvec, Jmatrix

from .....utils import Zero
from dask.distributed import get_client
import dask.array as da
import numpy as np
from scipy import sparse as sp
import zarr

import numcodecs

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

numcodecs.blosc.use_threads = False


def fields(self, m=None, return_Ainv=False):
    if m is not None:
        self.model = m

    if getattr(self, "_stashed_fields", None) is not None and not return_Ainv:
        return self._stashed_fields

    A = self.getA()
    Ainv = self.solver(A, **self.solver_opts)
    RHS = self.getRHS()

    f = self.fieldsPair(self)
    f[:, self._solutionType] = Ainv * np.asarray(RHS.todense())

    self._stashed_fields = f
    if return_Ainv:
        return f, Ainv
    return f


def compute_J(self, m, f=None):

    f, Ainv = self.fields(m=m, return_Ainv=True)

    m_size = m.size
    row_chunks = int(
        np.ceil(
            float(self.survey.nD)
            / np.ceil(float(m_size) * self.survey.nD * 8.0 * 1e-6 / self.max_chunk_size)
        )
    )

    if self.store_sensitivities == "disk":
        Jmatrix = zarr.open(
            self.sensitivity_path + "J.zarr",
            mode="w",
            shape=(self.survey.nD, m_size),
            chunks=(row_chunks, m_size),
        )
    else:
        Jmatrix = np.zeros((self.survey.nD, m_size), dtype=np.float32)

    blocks = []
    count = 0
    for source in self.survey.source_list:
        u_source = f[source, self._solutionType]

        for rx in source.receiver_list:

            if getattr(rx, "orientation", None) is not None:
                projected_grid = f._GLoc(rx.projField) + rx.orientation
            else:
                projected_grid = f._GLoc(rx.projField)

            PTv = rx.getP(self.mesh, projected_grid).toarray().T

            for dd in range(int(np.ceil(PTv.shape[1] / row_chunks))):
                start, end = dd * row_chunks, np.min(
                    [(dd + 1) * row_chunks, PTv.shape[1]]
                )
                df_duTFun = getattr(f, "_{0!s}Deriv".format(rx.projField), None)
                df_duT, df_dmT = df_duTFun(
                    source, None, PTv[:, start:end], adjoint=True
                )
                ATinvdf_duT = Ainv * df_duT
                dA_dmT = self.getADeriv(u_source, ATinvdf_duT, adjoint=True)
                dRHS_dmT = self.getRHSDeriv(source, ATinvdf_duT, adjoint=True)
                du_dmT = -dA_dmT
                if not isinstance(dRHS_dmT, Zero):
                    du_dmT += dRHS_dmT
                if not isinstance(df_dmT, Zero):
                    du_dmT += df_dmT

                #
                du_dmT = du_dmT.T.reshape((-1, m_size))

                if len(blocks) == 0:
                    blocks = du_dmT
                else:
                    blocks = np.vstack([blocks, du_dmT])

                while blocks.shape[0] >= row_chunks:

                    if self.store_sensitivities == "disk":
                        Jmatrix.set_orthogonal_selection(
                            (np.arange(count, count + row_chunks), slice(None)),
                            blocks[:row_chunks, :].astype(np.float32),
                        )
                    else:
                        Jmatrix[count : count + row_chunks, :] = blocks[
                            :row_chunks, :
                        ].astype(np.float32)

                    blocks = blocks[row_chunks:, :].astype(np.float32)
                    count += row_chunks

                del df_duT, ATinvdf_duT, dA_dmT, dRHS_dmT, du_dmT

    if len(blocks) != 0:

        if self.store_sensitivities == "disk":
            Jmatrix.set_orthogonal_selection(
                (np.arange(count, self.survey.nD), slice(None)),
                blocks.astype(np.float32),
            )
        else:
            Jmatrix[count : self.survey.nD, :] = blocks.astype(np.float32)

    Ainv.clean()

    if self.store_sensitivities == "disk":
        del Jmatrix
        self._Jmatrix = da.from_zarr(self.sensitivity_path + "J.zarr")
    else:
        self._Jmatrix = Jmatrix

    return self._Jmatrix


def source_eval(simulation, sources, indices):
    """
    Evaluate the source term for the given source and index
    """
    blocks = []
    for ind in indices:
        blocks.append(sources[ind].eval(simulation))

    return sp.csr_matrix(np.vstack(blocks).T)


def getSourceTerm(self):
    """
    Evaluates the sources, and puts them in matrix form
    :rtype: tuple
    :return: q (nC or nN, nSrc)
    """

    if getattr(self, "_q", None) is None:

        if self._mini_survey is not None:
            source_list = self._mini_survey.source_list
        else:
            source_list = self.survey.source_list

        indices = np.arange(len(source_list))
        try:

            client = get_client()
            sim = client.scatter(self, workers=self.worker)
            future_list = client.scatter(source_list, workers=self.worker)
            indices = np.array_split(indices, self.n_threads(client=client))
            blocks = []
            for ind in indices:
                blocks.append(
                    client.submit(
                        source_eval, sim, future_list, ind, workers=self.worker
                    )
                )

            blocks = sp.hstack(client.gather(blocks))
        except ValueError:
            blocks = source_eval(self, source_list, indices)

        self._q = blocks

    return self._q


Sim.getSourceTerm = getSourceTerm
Sim.fields = fields
Sim.compute_J = compute_J

Sim.getJtJdiag = getJtJdiag
Sim.Jvec = Jvec
Sim.Jtvec = Jtvec
Sim.Jmatrix = Jmatrix
