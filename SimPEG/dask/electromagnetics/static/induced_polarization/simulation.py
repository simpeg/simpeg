from .....electromagnetics.static.induced_polarization.simulation import (
    BaseIPSimulation as Sim,
)

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
    if self._Jmatrix is not None:
        return self._Jmatrix

    self.model = m
    if f is None:
        f = self.fields(m)

    if self.verbose:
        print("Calculating J and storing")

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
            PT = rx.getP(self.mesh, rx.projGLoc(f)).toarray().T
            df_duT = PT
            # Find a block of receivers
            n_block_col = int(np.ceil(df_duT.size * 8 * 1e-9 / self.max_ram))

            n_col = int(np.ceil(df_duT.shape[1] / n_block_col))

            nrows = int(
                m_size / np.ceil(m_size * n_col * 8 * 1e-6 / self.max_chunk_size)
            )
            ind = 0
            for col in range(n_block_col):
                ATinvdf_duT = da.asarray(
                    self.Ainv * df_duT[:, ind : ind + n_col]
                ).rechunk((nrows, n_col))
                dA_dmT = self.getADeriv(u_source, ATinvdf_duT, adjoint=True)
                # du_dmT = -da.from_delayed(dask.delayed(dA_dmT), shape=(self.model.size, n_col), dtype=float)
                if n_col > 1:
                    du_dmT = da.from_delayed(
                        dask.delayed(-dA_dmT), shape=(m_size, n_col), dtype=float
                    )
                else:
                    du_dmT = da.from_delayed(
                        dask.delayed(-dA_dmT), shape=(m_size,), dtype=float
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


def dask_getJtJdiag(self, m, W=None):
    """
        Return the diagonal of JtJ
    """
    if self.gtgdiag is None:

        # Need to check if multiplying weights makes sense
        if W is None:
            self.gtgdiag = da.sum(self.getJ(m) ** 2, axis=0).compute()
        else:
            w = da.from_array(W.diagonal())[:, None]
            self.gtgdiag = da.sum((w * self.getJ(m)) ** 2, axis=0).compute()

    return self.gtgdiag


Sim.getJtJdiag = dask_getJtJdiag
