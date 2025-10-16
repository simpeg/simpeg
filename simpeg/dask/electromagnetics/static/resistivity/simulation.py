from .....electromagnetics.static.resistivity.simulation import BaseDCSimulation as Sim
from .....utils import Zero

from ....utils import compute_chunk_sizes

import dask
import dask.array as da
import os
import shutil
import numpy as np


def dask_getJ(self, m, f=None):
    """
    Generate Full sensitivity matrix
    """

    if getattr(self, "_Jmatrix", None) is not None:
        return self._Jmatrix
    if f is None:
        f = self.fields(m)

    if self.verbose:
        print("Calculating J and storing")

    if self._mini_survey is not None:
        # Need to use _Jtvec for this operation currently...
        J = self._Jtvec(m=m, v=None, f=f).T
        self._Jmatrix = da.from_array(J)
        return self._Jmatrix

    if os.path.exists(self.sensitivity_path):
        shutil.rmtree(self.sensitivity_path, ignore_errors=True)

        # Wait for the system to clear out the directory
        while os.path.exists(self.sensitivity_path):
            pass

    m_size = self.model.size
    count = 0
    for source in self.survey.source_list:
        u_source = f[source, self._solutionType]
        for rx in source.receiver_list:
            # wrt f, need possibility wrt m
            PTv = rx.evalDeriv(source, self.mesh, f).toarray().T

            df_duTFun = getattr(f, "_{0!s}Deriv".format(rx.projField), None)
            df_duT, df_dmT = df_duTFun(source, None, PTv, adjoint=True)

            # Find a block of receivers
            n_block_col = int(np.ceil(df_duT.size * 8 * 1e-9 / self.max_ram))

            n_col = int(np.ceil(df_duT.shape[1] / n_block_col))

            nrows = int(
                m_size / np.ceil(m_size * n_col * 8 * 1e-6 / self.max_chunk_size)
            )
            ind = 0
            for _ in range(n_block_col):
                ATinvdf_duT = da.asarray(
                    self.Ainv * df_duT[:, ind : ind + n_col]
                ).rechunk((nrows, n_col))

                dA_dmT = self.getADeriv(u_source, ATinvdf_duT, adjoint=True)

                dRHS_dmT = self.getRHSDeriv(source, ATinvdf_duT, adjoint=True)

                if n_col > 1:
                    du_dmT = da.from_delayed(
                        dask.delayed(-dA_dmT), shape=(m_size, n_col), dtype=float
                    )
                else:
                    du_dmT = da.from_delayed(
                        dask.delayed(-dA_dmT), shape=(m_size,), dtype=float
                    )

                if not isinstance(dRHS_dmT, Zero):
                    du_dmT += da.from_delayed(
                        dask.delayed(dRHS_dmT), shape=(m_size, n_col), dtype=float
                    )

                if not isinstance(df_dmT, Zero):
                    du_dmT += da.from_delayed(
                        df_dmT, shape=(m_size, n_col), dtype=float
                    )

                blockName = self.sensitivity_path + "J" + str(count) + ".zarr"
                da.to_zarr((du_dmT.T).rechunk("auto"), blockName)
                del ATinvdf_duT
                count += 1

                ind += n_col

    dask_arrays = []
    for ii in range(count):
        blockName = self.sensitivity_path + "J" + str(ii) + ".zarr"
        J = da.from_zarr(blockName)
        # Stack all the source blocks in one big zarr
        dask_arrays.append(J)

    rowChunk, colChunk = compute_chunk_sizes(
        self.survey.nD, m_size, self.max_chunk_size
    )
    self._Jmatrix = da.vstack(dask_arrays).rechunk((rowChunk, colChunk))
    self.Ainv.clean()

    return self._Jmatrix


Sim.getJ = dask_getJ


def dask_getJtJdiag(self, m, W=None, f=None):
    """
    Return the diagonal of JtJ
    """
    if getattr(self, "_gtgdiag", None) is None:
        # Need to check if multiplying weights makes sense
        if W is None:
            self._gtgdiag = da.sum(self.getJ(m, f=f) ** 2, axis=0).compute()
        else:
            w = da.from_array(W.diagonal())[:, None]
            self._gtgdiag = da.sum((w * self.getJ(m, f=f)) ** 2, axis=0).compute()

    return self._gtgdiag


Sim.getJtJdiag = dask_getJtJdiag
