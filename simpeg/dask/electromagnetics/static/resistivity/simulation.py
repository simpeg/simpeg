from .....electromagnetics.static.resistivity.simulation import Simulation3DNodal as Sim
from ....simulation import BaseSimulation
from .....utils import Zero
import dask.array as da
import numpy as np
import zarr

import numcodecs

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

numcodecs.blosc.use_threads = False


class Simulation3DNodal(BaseSimulation, Sim):
    """
    Overload of the Simulation3DNodal to include the dask operations
    """

    def fields(self, m=None):
        if m is not None:
            self.model = m

        if getattr(self, "_stashed_fields", None) is not None:
            return self._stashed_fields

        A = self.getA()
        Ainv = self.solver(A, **self.solver_opts)
        RHS = self.getRHS()

        f = self.fieldsPair(self)
        f[:, self._solutionType] = Ainv * RHS

        self.Ainv = Ainv

        self._stashed_fields = f

        return f

    def compute_J(self, m, f=None):

        if f is None:
            f = self.fields(m)

        m_size = m.size
        row_chunks = int(
            np.ceil(
                float(self.survey.nD)
                / np.ceil(
                    float(m_size) * self.survey.nD * 8.0 * 1e-6 / self.max_chunk_size
                )
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

                if rx.orientation is not None:
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
                    ATinvdf_duT = self.Ainv * df_duT
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

        self.Ainv.clean()

        if self.store_sensitivities == "disk":
            del Jmatrix
            self._Jmatrix = da.from_zarr(self.sensitivity_path + "J.zarr")
        else:
            self._Jmatrix = Jmatrix

        return self._Jmatrix

    def getSourceTerm(self):
        """
        Evaluates the sources, and puts them in matrix form
        :rtype: tuple
        :return: q (nC or nN, nSrc)
        """

        if getattr(self, "_q", None) is None:

            if self._mini_survey is not None:
                Srcs = self._mini_survey.source_list
            else:
                Srcs = self.survey.source_list

            if self._formulation == "EB":
                n = self.mesh.nN
                # return NotImplementedError

            elif self._formulation == "HJ":
                n = self.mesh.nC

            q = np.zeros((n, len(Srcs)), order="F")

            for i, source in enumerate(Srcs):
                q[:, i] = source.eval(self)

            self._q = q

        return self._q
