import numpy as np
import scipy as sp
import properties
import shutil

from ....utils import mkvc, sdiag, Zero
from ....data import Data
from ...base import BaseEMSimulation
from .boundary_utils import getxBCyBC_CC
from .survey import Survey
from .fields import Fields_CC, Fields_N
import dask
import dask.array as da
import multiprocessing
import os
from scipy.sparse import csr_matrix as csr
from scipy.sparse import linalg
from pymatsolver import BicgJacobi
from pyMKL import mkl_set_num_threads, mkl_get_max_threads
import zarr
import time

class BaseDCSimulation(BaseEMSimulation):
    """
    Base DC Problem
    """
    Jpath = "./sensitivity/"
    # n_cpu = None
    maxRAM = 2

    survey = properties.Instance(
        "a DC survey object", Survey, required=True
    )

    storeJ = properties.Bool(
        "store the sensitivity matrix?", default=False
    )

    Ainv = None
    _Jmatrix = None
    gtgdiag = None
    n_cpu = int(multiprocessing.cpu_count())
    max_chunk_size = None

    @dask.delayed(pure=True)
    def fields(self, m=None, calcJ=True):

        mkl_set_num_threads(self.n_cpu)

        if m is not None:
            self.model = m
            self._Jmatrix = None

        f = self.fieldsPair(self)
        A = self.getA()

        A = self.getA()
        self.Ainv = self.Solver(A, **self.solver_opts)
        RHS = self.getRHS()
        Srcs = self.survey.source_list

        print("Fields n_cpu %i" % self.n_cpu)
        f[Srcs, self._solutionType] = self.Ainv * RHS  #, num_cores=self.n_cpu).compute()

        if not self.storeJ:
            self.Ainv.clean()

        return f

    def getJtJdiag(self, m, W=None):
        """
            Return the diagonal of JtJ
        """

        if (self.gtgdiag is None):

            # Need to check if multiplying weights makes sense
            if W is None:
                self.gtgdiag = da.sum((self.getJ(m))**2., 0).compute()
            else:
                self.gtgdiag = da.sum((self.getJ(m))**2., 0).compute()
            #     from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler

            #     with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
            #         self.gtgdiag = da.sum((self.getJ(m))**2., 0).compute()

            #     from dask.diagnostics import visualize
            #     visualize([prof, rprof, cprof])
                # WJ = da.from_delayed(
                #         dask.delayed(csr.dot)(W, self.getJ(m)),
                #         shape=self.getJ(m).shape,
                #         dtype=float
                # )
                # self.gtgdiag = da.sum(WJ**2., 0).compute()

        return self.gtgdiag

    def getJ(self, m, f=None):
        """
            Generate Full sensitivity matrix
        """

        if self._Jmatrix is not None:
            return self._Jmatrix
        else:

            self.model = m
            if f is None:
                f = self.fields(m).compute()

        if self.verbose:
            print("Calculating J and storing")

        if os.path.exists(self.Jpath):
            shutil.rmtree(self.Jpath, ignore_errors=True)

            # Wait for the system to clear out the directory
            while os.path.exists(self.Jpath):
                pass

        Jtv = []

        nD = self.survey.nD
        nC = m.shape[0]


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

        J = []

        tc = time.time()
        count = 0
        for source in self.survey.source_list:
            u_source = f[source, self._solutionType].copy()
            for rx in source.receiver_list:
                # wrt f, need possibility wrt m
                PTv = rx.getP(self.mesh, rx.projGLoc(f)).T

                df_duTFun = getattr(f, '_{0!s}Deriv'.format(rx.projField),
                                    None)
                df_duT, df_dmT = df_duTFun(source, None, PTv, adjoint=True)

                # Find a block of receivers
                n_block_col = int(np.ceil(df_duT.shape[0]*df_duT.shape[1]*8*1e-9 / self.maxRAM))

                n_col = int(np.ceil(df_duT.shape[1] / n_block_col))

                ind = 0
                for col in range(n_block_col):
                    ATinvdf_duT = da.asarray(self.Ainv * np.asarray(df_duT[:, ind:ind+n_col].todense())).rechunk((df_duT.shape[0], 1))

                    stack = []
                    for v in range(ATinvdf_duT.shape[1]):

                        vec = ATinvdf_duT[:, v].reshape((df_duT.shape[0], 1))
                        # if len(ATinvdf_duT.shape) == 1:
                        #     ATinvdf_duT =

                        dA_dmT = self.getADeriv(u_source, vec, adjoint=True)

                        dRHS_dmT = self.getRHSDeriv(source, vec, adjoint=True)

                        du_dmT = -dA_dmT

                        if not isinstance(dRHS_dmT, Zero):
                            du_dmT += da.from_delayed(da.delayed(dRHS_dmT), shape=(self.model.size, n_col), dtype=float)

                        if not isinstance(df_dmT, Zero):

                            du_dmT += da.from_delayed(df_dmT, shape=(self.model.size, n_col), dtype=float)

                        stack += [du_dmT]

                    blockName = self.Jpath + "J" + str(count) + ".zarr"

                    da.to_zarr((da.hstack(stack).T).rechunk('auto'), blockName)
                    count += 1

                    ind += n_col
                    # Jtv.append(du_dmT)


        print("Block time: %f"%(time.time()-tc))
        dask_arrays = []
        for ii in range(count):
            blockName = self.Jpath + "J" + str(ii) + ".zarr"
            J = da.from_zarr(blockName)
        # Stack all the source blocks in one big zarr
            dask_arrays.append(J)

            # J = da.hstack(Jtv).T
            # J = J.rechunk((rowChunk, colChunk))

        self._Jmatrix = da.concatenate(dask_arrays, axis=0).rechunk((rowChunk, colChunk))
        print("Addpen time: %f"%(time.time()-tc))
        self.Ainv.clean()

        return self._Jmatrix

    def Jvec(self, m, v, f=None):
        """
            Compute sensitivity matrix (J) and vector (v) product.
        """

        self.model = m

        if f is None:
            f = self.fields(m)

        if self.storeJ:

            J = self.getJ(m, f=f)
            return da.dot(J, da.from_array(v, chunks=self._Jmatrix.chunks[1]))

        Jv = []
        for source in self.survey.source_list:
            u_source = f[source, self._solutionType]  # solution vector
            dA_dm_v = self.getADeriv(u_source, v).compute()
            dRHS_dm_v = self.getRHSDeriv(source, v).compute()
            du_dm_v = self.Ainv * (- dA_dm_v + dRHS_dm_v)
            for rx in source.receiver_list:
                df_dmFun = getattr(f, '_{0!s}Deriv'.format(rx.projField), None)
                df_dm_v = df_dmFun(source, du_dm_v, v, adjoint=False)
                Jv.append(rx.evalDeriv(source, self.mesh, f, df_dm_v))
        return np.hstack(Jv)

    def Jtvec(self, m, v, f=None):
        """
            Compute adjoint sensitivity matrix (J^T) and vector (v) product.

        """

        if f is None:
            f = self.fields(m)

        self.model = m

        if self.storeJ:

            J = self.getJ(m, f=f)

            return mkvc(
                da.dot(
                    da.from_array(
                        v, chunks=self._Jmatrix.chunks[0]), self._Jmatrix
                    ).compute()
                )

        return self._Jtvec(m, v=v, f=f)

    def _Jtvec(self, m, v=None, f=None):
        """
            Compute adjoint sensitivity matrix (J^T) and vector (v) product.
            Full J matrix can be computed by inputing v=None
        """

        if v is not None:
            # Ensure v is a data object.
            if not isinstance(v, Data):
                v = Data(self.survey, v)
            Jtv = np.zeros(m.size)
        else:
            # This is for forming full sensitivity matrix
            Jtv = np.zeros((self.model.size, self.survey.nD), order='F')
            istrt = int(0)
            iend = int(0)


        for source in self.survey.source_list:
            u_source = f[source, self._solutionType].copy()
            for rx in source.receiver_list:
                # wrt f, need possibility wrt m
                if v is not None:
                    PTv = rx.evalDeriv(
                        source, self.mesh, f, v[source, rx], adjoint=True
                    )
                else:
                    # This is for forming full sensitivity matrix
                    PTv = rx.getP(self.mesh, rx.projGLoc(f)).toarray().T
                df_duTFun = getattr(f, '_{0!s}Deriv'.format(rx.projField),
                                    None)
                df_duT, df_dmT = df_duTFun(source, None, PTv, adjoint=True)

                ATinvdf_duT = self.Ainv * df_duT

                dA_dmT = self.getADeriv(u_source, ATinvdf_duT, adjoint=True).compute()
                dRHS_dmT = self.getRHSDeriv(source, ATinvdf_duT, adjoint=True).compute()
                du_dmT = -dA_dmT + dRHS_dmT
                if v is not None:
                    Jtv += (df_dmT + du_dmT).astype(float)
                else:
                    iend = istrt + rx.nD
                    if rx.nD == 1:
                        Jtv[:, istrt] = (df_dmT + du_dmT)
                    else:
                        Jtv[:, istrt:iend] = (df_dmT + du_dmT)
                    istrt += rx.nD

        if v is not None:
            return mkvc(Jtv)
        else:
            # return np.hstack(Jtv)
            return Jtv

    def getSourceTerm(self):
        """
        Evaluates the sources, and puts them in matrix form
        :rtype: tuple
        :return: q (nC or nN, nSrc)
        """

        Srcs = self.survey.source_list

        if self._formulation == 'EB':
            n = self.mesh.nN
            # return NotImplementedError

        elif self._formulation == 'HJ':
            n = self.mesh.nC

        q = np.zeros((n, len(Srcs)))

        for i, source in enumerate(Srcs):
            q[:, i] = source.eval(self)
        return q

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = super(BaseDCSimulation, self).deleteTheseOnModelUpdate
        if self._Jmatrix is not None:
            toDelete += ['_Jmatrix']
        return toDelete


class Problem3D_CC(BaseDCSimulation):
    """
    3D cell centered DC problem
    """

    _solutionType = 'phiSolution'
    _formulation = 'HJ'  # CC potentials means J is on faces
    fieldsPair = Fields_CC
    bc_type = 'Dirichlet'

    def __init__(self, mesh, **kwargs):

        BaseDCSimulation.__init__(self, mesh, **kwargs)
        self.setBC()

    def getA(self):
        """
        Make the A matrix for the cell centered DC resistivity problem
        A = D MfRhoI G
        """

        D = self.Div
        G = self.Grad
        MfRhoI = self.MfRhoI
        A = D * MfRhoI * G

        if(self.bc_type == 'Neumann'):
            Vol = self.mesh.vol
            if self.verbose:
                print('Perturbing first row of A to remove nullspace for Neumann BC.')

            # Handling Null space of A
            I, J, V = sp.sparse.find(A[0, :])
            for jj in J:
                A[0, jj] = 0.
            A[0, 0] = 1.

        # I think we should deprecate this for DC problem.
        # if self._makeASymmetric is True:
        #     return V.T * A
        return A

    def getADeriv(self, u, v, adjoint=False):

        # Gvec = da.from_delayed(
        #     dask.delayed(csr.dot)(self.Grad, u),
        #     dtype=float, shape=[self.Grad.shape[0]]
        # )
        Gvec = self.Grad * u
        MfRhoIDeriv = self.MfRhoIDeriv
        if adjoint:
            Dvec = da.from_delayed(
                dask.delayed(csr.dot)(self.Div.T, v),
                dtype=float, shape=[self.Div.shape[1], v.shape[1]]
            )
            return MfRhoIDeriv(Gvec, Dvec, adjoint)

        vec = MfRhoIDeriv(Gvec, v, adjoint)

        Dvec = da.from_delayed(
            dask.delayed(csr.dot)(self.Div, vec),
            dtype=float, shape=[self.Div.shape[0], vec.shape[1]]
        )

        return Dvec

    def getRHS(self):
        """
        RHS for the DC problem
        q
        """

        RHS = self.getSourceTerm()

        return RHS

    def getRHSDeriv(self, source, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = source.evalDeriv(self, adjoint=adjoint)
        # return qDeriv
        return Zero()

    def MfRhoIDeriv(self, u, v=None, adjoint=False):
        """
            Derivative of :code:`MfRhoI` with respect to the model.
        """
        if self.rhoMap is None:
            return Zero()

        if len(self.rho.shape) > 1:
            if self.rho.shape[1] > self.mesh.dim:
                raise NotImplementedError(
                    "Full anisotropy is not implemented for MfRhoIDeriv."
                )

        if getattr(self, 'dMfRhoI_dI', None) is None:
            self.dMfRhoI_dI = -self.MfRhoI.power(2.)

        if adjoint is True:
#            vec = da.from_delayed(
#                dask.delayed(csr.dot)(self.dMfRhoI_dI.T, u),
#                dtype=float, shape=[self.dMfRhoI_dI.shape[1]]
#            )
            vec = self.dMfRhoI_dI.T * u
            return self.MfRhoDeriv(
                vec, v=v, adjoint=adjoint
            )
        else:
            vec = self.MfRhoDeriv(u, v=v)

            return da.from_delayed(
                dask.delayed(csr.dot)(self.dMfRhoI_dI, vec),
                dtype=float, shape=[self.dMfRhoI_dI.shape[0], vec.shape[1]]
            )
            # return dMfRhoI_dI * self.MfRhoDeriv(u, v=v)

    def MfRhoDeriv(self, u, v=None, adjoint=False):
        """
        Derivative of :code:`MfRho` with respect to the model.
        """
        if self.rhoMap is None:
            return Zero()

        if getattr(self, '_MfRhoDeriv', None) is None:
            self._MfRhoDeriv = self.mesh.getFaceInnerProductDeriv(
                np.ones(self.mesh.nC)
            )(np.ones(self.mesh.nF)) * self.rhoDeriv

        if v is not None:
            if adjoint is True:
                vec = da.from_delayed(
                    dask.delayed(csr.dot)(sdiag(u), v),
                    dtype=float, shape=[u.shape[0], v.shape[1]]
                )
                return da.from_delayed(
                    dask.delayed(csr.dot)(self._MfRhoDeriv.T, vec),
                    dtype=float, shape=[self._MfRhoDeriv.shape[1], vec.shape[1]]
                )

            vec = da.from_delayed(
                dask.delayed(csr.dot)(self._MfRhoDeriv, v),
                dtype=float, shape=[self._MfRhoDeriv.shape[0], v.shape[1]]
            )
            return da.from_delayed(
                    dask.delayed(csr.dot)(sdiag(u), vec),
                    dtype=float, shape=[u.shape[0], vec.shape[1]]
                )
        else:
            if adjoint is True:

                return da.from_delayed(
                    dask.delayed(csr.dot)(self._MfRhoDeriv.T, sdiag(u)),
                    dtype=float, shape=[self._MfRhoDeriv.shape[1], u.shape[0]]
                )

            return da.from_delayed(
                    dask.delayed(csr.dot)(sdiag(u), self._MfRhoDeriv),
                    dtype=float, shape=[u.shape[0], self._MfRhoDeriv.shape[1]]
                )

            # sdiag(u)*(self._MfRhoDeriv)

    def setBC(self):
        if self.bc_type == 'Dirichlet':
            print('Homogeneous Dirichlet is the natural BC for this CC discretization.')
            self.Div = sdiag(self.mesh.vol) * self.mesh.faceDiv
            self.Grad = self.Div.T

        else:
            if self.mesh._meshType == "TREE" and self.bc_type == 'Neumann':
                raise NotImplementedError()

            if self.mesh.dim == 3:
                fxm, fxp, fym, fyp, fzm, fzp = self.mesh.faceBoundaryInd
                gBFxm = self.mesh.gridFx[fxm, :]
                gBFxp = self.mesh.gridFx[fxp, :]
                gBFym = self.mesh.gridFy[fym, :]
                gBFyp = self.mesh.gridFy[fyp, :]
                gBFzm = self.mesh.gridFz[fzm, :]
                gBFzp = self.mesh.gridFz[fzp, :]

                # Setup Mixed B.C (alpha, beta, gamma)
                temp_xm = np.ones_like(gBFxm[:, 0])
                temp_xp = np.ones_like(gBFxp[:, 0])
                temp_ym = np.ones_like(gBFym[:, 1])
                temp_yp = np.ones_like(gBFyp[:, 1])
                temp_zm = np.ones_like(gBFzm[:, 2])
                temp_zp = np.ones_like(gBFzp[:, 2])

                if(self.bc_type == 'Neumann'):
                    if self.verbose:
                        print('Setting BC to Neumann.')
                    alpha_xm, alpha_xp = temp_xm*0., temp_xp*0.
                    alpha_ym, alpha_yp = temp_ym*0., temp_yp*0.
                    alpha_zm, alpha_zp = temp_zm*0., temp_zp*0.

                    beta_xm, beta_xp = temp_xm, temp_xp
                    beta_ym, beta_yp = temp_ym, temp_yp
                    beta_zm, beta_zp = temp_zm, temp_zp

                    gamma_xm, gamma_xp = temp_xm*0., temp_xp*0.
                    gamma_ym, gamma_yp = temp_ym*0., temp_yp*0.
                    gamma_zm, gamma_zp = temp_zm*0., temp_zp*0.

                elif(self.bc_type == 'Dirichlet'):
                    if self.verbose:
                        print('Setting BC to Dirichlet.')
                    alpha_xm, alpha_xp = temp_xm, temp_xp
                    alpha_ym, alpha_yp = temp_ym, temp_yp
                    alpha_zm, alpha_zp = temp_zm, temp_zp

                    beta_xm, beta_xp = temp_xm*0, temp_xp*0
                    beta_ym, beta_yp = temp_ym*0, temp_yp*0
                    beta_zm, beta_zp = temp_zm*0, temp_zp*0

                    gamma_xm, gamma_xp = temp_xm*0., temp_xp*0.
                    gamma_ym, gamma_yp = temp_ym*0., temp_yp*0.
                    gamma_zm, gamma_zp = temp_zm*0., temp_zp*0.

                elif(self.bc_type == 'Mixed'):
                    # Ztop: Neumann
                    # Others: Mixed: alpha * phi + d phi dn = 0
                    # where alpha = 1 / r  * dr/dn
                    # (Dey and Morrison, 1979)

                    # This assumes that the source is located at
                    # (x_center, y_center_y, ztop)
                    # TODO: Implement Zhang et al. (1995)

                    xs = np.median(self.mesh.vectorCCx)
                    ys = np.median(self.mesh.vectorCCy)
                    zs = self.mesh.vectorCCz[-1]

                    def r_boundary(x, y, z):
                        return 1./np.sqrt(
                            (x - xs)**2 + (y - ys)**2 + (z - zs)**2
                        )
                    rxm = r_boundary(gBFxm[:, 0], gBFxm[:, 1], gBFxm[:, 2])
                    rxp = r_boundary(gBFxp[:, 0], gBFxp[:, 1], gBFxp[:, 2])
                    rym = r_boundary(gBFym[:, 0], gBFym[:, 1], gBFym[:, 2])
                    ryp = r_boundary(gBFyp[:, 0], gBFyp[:, 1], gBFyp[:, 2])
                    rzm = r_boundary(gBFzm[:, 0], gBFzm[:, 1], gBFzm[:, 2])

                    alpha_xm = (gBFxm[:, 0]-xs)/rxm**2
                    alpha_xp = (gBFxp[:, 0]-xs)/rxp**2
                    alpha_ym = (gBFym[:, 1]-ys)/rym**2
                    alpha_yp = (gBFyp[:, 1]-ys)/ryp**2
                    alpha_zm = (gBFzm[:, 2]-zs)/rzm**2
                    alpha_zp = temp_zp.copy() * 0.

                    beta_xm, beta_xp = temp_xm*1, temp_xp*1
                    beta_ym, beta_yp = temp_ym*1, temp_yp*1
                    beta_zm, beta_zp = temp_zm*1, temp_zp*1

                    gamma_xm, gamma_xp = temp_xm*0., temp_xp*0.
                    gamma_ym, gamma_yp = temp_ym*0., temp_yp*0.
                    gamma_zm, gamma_zp = temp_zm*0., temp_zp*0.

                alpha = [
                    alpha_xm, alpha_xp,
                    alpha_ym, alpha_yp,
                    alpha_zm, alpha_zp
                ]
                beta = [beta_xm, beta_xp, beta_ym, beta_yp, beta_zm, beta_zp]
                gamma = [gamma_xm, gamma_xp, gamma_ym, gamma_yp, gamma_zm,
                         gamma_zp]

            elif self.mesh.dim == 2:

                fxm, fxp, fym, fyp = self.mesh.faceBoundaryInd
                gBFxm = self.mesh.gridFx[fxm, :]
                gBFxp = self.mesh.gridFx[fxp, :]
                gBFym = self.mesh.gridFy[fym, :]
                gBFyp = self.mesh.gridFy[fyp, :]

                # Setup Mixed B.C (alpha, beta, gamma)
                temp_xm = np.ones_like(gBFxm[:, 0])
                temp_xp = np.ones_like(gBFxp[:, 0])
                temp_ym = np.ones_like(gBFym[:, 1])
                temp_yp = np.ones_like(gBFyp[:, 1])

                alpha_xm, alpha_xp = temp_xm*0., temp_xp*0.
                alpha_ym, alpha_yp = temp_ym*0., temp_yp*0.

                beta_xm, beta_xp = temp_xm, temp_xp
                beta_ym, beta_yp = temp_ym, temp_yp

                gamma_xm, gamma_xp = temp_xm*0., temp_xp*0.
                gamma_ym, gamma_yp = temp_ym*0., temp_yp*0.

                alpha = [alpha_xm, alpha_xp, alpha_ym, alpha_yp]
                beta = [beta_xm, beta_xp, beta_ym, beta_yp]
                gamma = [gamma_xm, gamma_xp, gamma_ym, gamma_yp]

            x_BC, y_BC = getxBCyBC_CC(self.mesh, alpha, beta, gamma)
            V = self.Vol
            self.Div = V * self.mesh.faceDiv
            P_BC, B = self.mesh.getBCProjWF_simple()
            M = B*self.mesh.aveCC2F
            self.Grad = self.Div.T - P_BC*sdiag(y_BC)*M


class Problem3D_N(BaseDCSimulation):
    """
    3D nodal DC problem
    """

    _solutionType = 'phiSolution'
    _formulation = 'EB'  # N potentials means B is on faces
    fieldsPair = Fields_N

    def __init__(self, mesh, **kwargs):
        BaseDCSimulation.__init__(self, mesh, **kwargs)
        # Not sure why I need to do this
        # To evaluate mesh.aveE2CC, this is required....
        if mesh._meshType == "TREE":
            mesh.nodalGrad

    def getA(self):
        """
        Make the A matrix for the cell centered DC resistivity problem
        A = G.T MeSigma G
        """

        MeSigma = self.MeSigma
        Grad = self.mesh.nodalGrad
        A = Grad.T * MeSigma * Grad

        # Handling Null space of A
        I, J, V = sp.sparse.find(A[0, :])
        for jj in J:
            A[0, jj] = 0.
        A[0, 0] = 1.

        return A

    @dask.delayed()
    def getADeriv(self, u, v, adjoint=False):
        """
        Product of the derivative of our system matrix with respect to the
        model and a vector
        """
        Grad = self.mesh.nodalGrad
        if not adjoint:
            return Grad.T*self.MeSigmaDeriv(Grad*u, v, adjoint)
        elif adjoint:
            return self.MeSigmaDeriv(Grad*u, Grad*v, adjoint)

    def getRHS(self):
        """
        RHS for the DC problem
        q
        """

        RHS = self.getSourceTerm()
        return RHS

    def getRHSDeriv(self, source, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = source.evalDeriv(self, adjoint=adjoint)
        # return qDeriv
        return Zero()
