from ....electromagnetics.static.resistivity import (
    BaseDCSimulation as Sim, Simulation3DCellCentered as SimCC
)
from ....utils import Zero
from ....electromagnetics.static.resistivity.utils import _mini_pole_pole

import dask
import dask.array as da
import os
import shutil
import numpy as np
import sparse

Sim.maxRam = 2
Sim.max_chunk_size = 128


def dask_getJ(self, m, f=None):
    """
        Generate Full sensitivity matrix
    """

    if self._Jmatrix is not None:
        return self._Jmatrix
    else:
        self.model = m
        if f is None:
            A = self.getA()
            self.Ainv = self.Solver(A, **self.solver_opts)

    if self.verbose:
        print("Calculating J and storing")

    if self._mini_survey is not None:
        # Need to us _Jtvec for this operation currently...
        J = self._Jtvec(m=m, v=None, f=f).T
        self._Jmatrix = da.from_array(J)
        return self._Jmatrix

    if os.path.exists(self.Jpath):
        shutil.rmtree(self.Jpath, ignore_errors=True)

        # Wait for the system to clear out the directory
        while os.path.exists(self.Jpath):
            pass

    def J_func(src_list, rx_list_per_source):
        for source, rx_list in zip(src_list, rx_list_per_source):
            if f is not None:
                u_source = f[source, self._solutionType]
            else:
                u_source = self.Ainv*source.eval(self)
            PT_src = np.empty(np.sum([rx.nD for rx in rx_list]), len(m), order='F')
            start = 0
            for rx in rx_list:
                # wrt f, need possibility wrt m
                PTv = rx.getP(self.mesh, rx.projGLoc(f)).toarray().T

                df_duTFun = getattr(f, '_{0!s}Deriv'.format(rx.projField),
                                    None)
                df_duT, df_dmT = df_duTFun(source, None, PTv, adjoint=True)
                end = start+rx.nD
                PT_src[start:end] = (df_dmT + du_dmT).reshape((rx.nD, len(m)), order='F')
                start = end

            ATinvdf_duT = self.Ainv * PT_src

            dA_dmT = self.getADeriv(u_source, ATinvdf_duT, adjoint=True)
            dRHS_dmT = self.getRHSDeriv(source, ATinvdf_duT, adjoint=True)

    count = 0
    for source in self.survey.source_list:
        if f is not None:
            u_source = f[source, self._solutionType]
        else:
            u_source = self.Ainv*source.eval(self)
        for rx in source.receiver_list:
            # wrt f, need possibility wrt m
            PTv = rx.getP(self.mesh, rx.projGLoc(f)).T

            df_duTFun = getattr(f, '_{0!s}Deriv'.format(rx.projField),
                                None)
            df_duT, df_dmT = df_duTFun(source, None, PTv, adjoint=True)

            # Find a block of receivers
            n_block_col = int(np.ceil(df_duT.shape[0]*df_duT.shape[1]*8*1e-9 / self.maxRAM))

            n_col = int(np.ceil(df_duT.shape[1] / n_block_col))

            nrows = int(self.model.size / np.ceil(self.model.size * n_col * 8 * 1e-6 / self.max_chunk_size))
            ind = 0
            for col in range(n_block_col):
                ATinvdf_duT = da.asarray(self.Ainv * np.asarray(df_duT[:, ind:ind+n_col].todense())).rechunk((nrows, n_col))

                dA_dmT = self.getADeriv(u_source, ATinvdf_duT, adjoint=True)

                dRHS_dmT = self.getRHSDeriv(source, ATinvdf_duT, adjoint=True)

                if n_col > 1:
                    du_dmT = da.from_delayed(dask.delayed(-dA_dmT),
                                             shape=(self.model.size, n_col),
                                             dtype=float)
                else:
                    du_dmT = da.from_delayed(dask.delayed(-dA_dmT),
                                             shape=(self.model.size,),
                                             dtype=float)

                if not isinstance(dRHS_dmT, Zero):
                    du_dmT += da.from_delayed(dask.delayed(dRHS_dmT), shape=(self.model.size, n_col), dtype=float)

                if not isinstance(df_dmT, Zero):
                    du_dmT += da.from_delayed(df_dmT, shape=(self.model.size, n_col), dtype=float)

                blockName = self.Jpath + "J" + str(count) + ".zarr"
                da.to_zarr((du_dmT.T).rechunk('auto'), blockName)
                del ATinvdf_duT
                count += 1

                ind += n_col

    dask_arrays = []
    for ii in range(count):
        blockName = self.Jpath + "J" + str(ii) + ".zarr"
        J = da.from_zarr(blockName)
        # Stack all the source blocks in one big zarr
        dask_arrays.append(J)

    rowChunk = self.rowChunk
    colChunk = self.colChunk
    self._Jmatrix = da.vstack(dask_arrays).rechunk((rowChunk, colChunk))
    self.Ainv.clean()

    return self._Jmatrix
Sim.getJ = dask_getJ


def dask_getJtJdiag(self, m, W=None):
    """
        Return the diagonal of JtJ
    """
    if (self.gtgdiag is None):

        # Need to check if multiplying weights makes sense
        if W is None:
            self.gtgdiag = da.sum(self.getJ(m)**2, axis=0).compute()
        else:
            w = da.from_array(W.diagonal())[:, None]
            self.gtgdiag = da.sum((w*self.getJ(m))**2, axis=0).compute()

    return self.gtgdiag
Sim.getJtJdiag = dask_getJtJdiag


def dask_sim_init(self, mesh, **kwargs):
    miniaturize = kwargs.pop('miniaturize', False)
    super(Sim, self).__init__(mesh, **kwargs)  # must call super this way here
    if miniaturize:
        self._dipoles, self._invs, self._mini_survey = _mini_pole_pole(self.survey)

    nD = self.survey.nD
    if self.sigmaMap is not None:
        nC = self.sigmaMap.shape[1]
    elif self.rhoMap is not None:
        nC = self.rhoMap.shape[1]
    else:
        raise ValueError('Either rhoMap or sigmaMap must be set')

    # print('DASK: Chunking using parameters')
    nChunks_col = 1
    nChunks_row = 1
    rowChunk = int(np.ceil(nD/nChunks_row))
    colChunk = int(np.ceil(nC/nChunks_col))
    chunk_size = rowChunk*colChunk*8*1e-6  # in Mb

    # Add more chunks until memory falls below target
    while chunk_size >= self.max_chunk_size:
        if rowChunk > colChunk:
            nChunks_row += 1
        else:
            nChunks_col += 1

        rowChunk = int(np.ceil(nD/nChunks_row))
        colChunk = int(np.ceil(nC/nChunks_col))
        chunk_size = rowChunk*colChunk*8*1e-6  # in Mb
    self.rowChunk = rowChunk
    self.colChunk = colChunk
Sim.__init__ = dask_sim_init


def dask_simCC_init(self, mesh, **kwargs):
    Sim.__init__(self, mesh, **kwargs)
    self.setBC()

    self.Div = da.from_array(
        sparse.COO.from_scipy_sparse(self.Div.T),
        chunks=(self.rowChunk, self.rowChunk), asarray=False
    )
SimCC.__init__ = dask_simCC_init
