import numpy as np
import scipy as sp
import properties
import shutil

from ....utils import mkvc, sdiag, Zero, diagEst
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
from pyMKL import mkl_set_num_threads
from pyMKL import mkl_get_max_threads

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
    n_cpu = 1

    def fields(self, m=None):
        if m is not None:
            self.model = m

        if self.Ainv is not None:
            self.Ainv.clean()

        if self._Jmatrix is not None:
            self._Jmatrix = None

        f = self.fieldsPair(self)
        A = self.getA()

        self.Ainv = self.Solver(A, **self.solver_opts)
        RHS = self.getRHS()
        # AinvRHS = dask.delayed(self.Ainv._solve)(RHS)
        # u = da.from_delayed(AinvRHS, shape=(A.shape[0], RHS.shape[1]), dtype=float)
        Srcs = self.survey.source_list
        f[Srcs, self._solutionType] = self.Ainv * RHS
        return f

    def getJtJdiag(self, m, W=None):
        """
            Return the diagonal of JtJ
        """
        # def JtJv(v):
        #     Jv = self.Jvec(m, v)
        #     return self.Jtvec(m, Jv)
        # if (self.gtgdiag is None):
        #     jtjdiag = diagEst(JtJv, len(m), k=10)
        #     jtjdiag = jtjdiag / np.max(jtjdiag)
        #     self.gtgdiag = jtjdiag
        # return self.gtgdiag

        if (self.gtgdiag is None):

            # Need to check if multiplying weights makes sense
            if W is None:
                self.gtgdiag = da.sum((self.getJ(self.model))**2., 0).compute()
            else:
                # WJ = da.from_delayed(
                #         dask.delayed(csr.dot)(W, self.getJ(self.model)),
                #         shape=self.getJ(self.model).shape,
                #         dtype=float
                # )
                # self.gtgdiag = da.sum(WJ**2., 0).compute()
                self.gtgdiag = da.sum(self._JtJdiag(self.model, W), 0).compute()
        return self.gtgdiag

    @dask.delayed(pure=True)
    def AinvXvec(self, v, num_cores=1):
        mkl_set_num_threads(num_cores)
        A = self.getA()
        Ainv = self.Solver(A, **self.solver_opts)
        _ainv_v = Ainv * v
        Ainv.clean()
        return _ainv_v

    def _JtJdiag(self, m, w, f=None):
        """
            Generate Full sensitivity matrix
        """
        print('_jtj')
        if self._Jmatrix is not None:
            return self._Jmatrix
        else:

            self.model = m
            if f is None:
                f = self.fields(m)

        if self.verbose:
            print("Calculating J and storing")

        if os.path.exists(self.Jpath):
            shutil.rmtree(self.Jpath, ignore_errors=True)

            # Wait for the system to clear out the directory
            while os.path.exists(self.Jpath):
                pass
        istrt = int(0)
        iend = int(0)
        # if os.path.exists(self.Jpath + "J.zarr"):
        #     self._Jmatrix = da.from_zarr(self.Jpath + "J.zarr")
        # else:
        self.n_cpu = int(multiprocessing.cpu_count())
        Jtv = []
        count = 0
        for source in self.survey.source_list:
            u_source = f[source, self._solutionType].copy()
            for rx in source.receiver_list:
                # wrt f, need possibility wrt m
                PTv = rx.getP(self.mesh, rx.projGLoc(f)).toarray().T

                df_duTFun = getattr(f, '_{0!s}Deriv'.format(rx.projField),
                                    None)
                df_duT, df_dmT = df_duTFun(source, None, PTv, adjoint=True)

                # Compute block of receivers
                ATinvdf_duT = self.Ainv * df_duT

                if len(ATinvdf_duT.shape) == 1:
                    ATinvdf_duT = np.c_[ATinvdf_duT]

                dA_dmT = self.getADeriv(u_source, ATinvdf_duT, adjoint=True)

                dRHS_dmT = self.getRHSDeriv(source, ATinvdf_duT, adjoint=True)

                du_dmT = -da.from_delayed(dA_dmT, shape=(self.model.size, rx.nD), dtype=float) + da.from_delayed(dRHS_dmT, shape=(self.model.size, rx.nD), dtype=float)

                if not isinstance(df_dmT, Zero):
                    du_dmT += da.from_delayed(df_dmT, shape=(self.model.size, rx.nD), dtype=float)

                blockName = self.Jpath + "J" + str(count) + ".zarr"
                nChunks = self.n_cpu  # Number of chunks
                rowChunk = int(np.ceil(rx.nD/nChunks))
                colChunk = int(np.ceil(self.model.size/nChunks))  # Chunk sizes
                du_dmT = du_dmT.rechunk((colChunk, rowChunk))

                da.to_zarr(du_dmT, blockName)
                # here we apply the weights
                iend = istrt + rx.nD
                w_ = w[istrt:iend, istrt:iend]
                WJ = da.from_delayed(
                    dask.delayed(csr.dot)(du_dmT, w_),
                    shape=du_dmT.shape,
                    dtype=float
                )
                # print(da.sum(WJ**2, 1).shape)
                Jtv.append(da.sum(WJ**2, 1))
                istrt += rx.nD
                count += 1

        # Stack all the summed source blocks in one big zarr
        J = da.vstack(Jtv)
        nChunks = self.n_cpu  # Number of chunks
        rowChunk = int(np.ceil(count/nChunks))# Chunk sizes
        colChunk = int(np.ceil(m.shape[0]/nChunks))
        J = J.rechunk((rowChunk, colChunk))

        da.to_zarr(J, self.Jpath + "J.zarr")
        # self._Jmatrix = da.from_zarr(self.Jpath + "J.zarr")
        return da.from_zarr(self.Jpath + "J.zarr")

    def getJ2(self, m, f=None):
        """
            Generate Full sensitivity matrix
        """

        if self._Jmatrix is not None:
            return self._Jmatrix
        else:

            self.model = m
            if f is None:
                f = self.fields(m)

        if self.verbose:
            print("Calculating J and storing")

        Jtv = []
        count = 0
        # print('J2')
        for source in self.survey.source_list:
            u_source = f[source, self._solutionType].copy()
            for rx in source.receiver_list:
                # wrt f, need possibility wrt m
                PTv = rx.getP(self.mesh, rx.projGLoc(f)).toarray().T

                df_duTFun = getattr(f, '_{0!s}Deriv'.format(rx.projField),
                                    None)
                df_duT, df_dmT = df_duTFun(source, None, PTv, adjoint=True)

                # Compute block of receivers
                ATinvdf_duT = da.from_delayed(self.AinvXvec(df_duT, num_cores=self.n_cpu), shape=(self.model.size, rx.nD), dtype=float)

                dA_dmT = self.getADeriv(u_source, ATinvdf_duT, adjoint=True)

                dRHS_dmT = self.getRHSDeriv(source, ATinvdf_duT, adjoint=True)

                du_dmT = -da.from_delayed(dA_dmT, shape=(self.model.size, rx.nD), dtype=float) + da.from_delayed(dRHS_dmT, shape=(self.model.size, rx.nD), dtype=float)

                if not isinstance(df_dmT, Zero):
                    du_dmT += da.from_delayed(df_dmT, shape=(self.model.size, rx.nD), dtype=float)

                Jtv.append(du_dmT)
                count += 1

        # clean all factorization
        if self.Ainv is not None:
            self.Ainv.clean()
        # Stack all the sources
        J = da.hstack(Jtv).T
        self._Jmatrix = J

        return self._Jmatrix

    def getJ(self, m, f=None):
        """
            Generate Full sensitivity matrix
        """

        if self._Jmatrix is not None:
            return self._Jmatrix
        else:

            self.model = m
            if f is None:
                f = self.fields(m)

        if self.verbose:
            print("Calculating J and storing")

        if os.path.exists(self.Jpath):
            shutil.rmtree(self.Jpath, ignore_errors=True)

            # Wait for the system to clear out the directory
            while os.path.exists(self.Jpath):
                pass

        # if os.path.exists(self.Jpath + "J.zarr"):
        #     self._Jmatrix = da.from_zarr(self.Jpath + "J.zarr")
        # else:
        self.n_cpu = int(multiprocessing.cpu_count())
        Jtv = []
        count = 0
        for source in self.survey.source_list:
            u_source = f[source, self._solutionType].copy()
            for rx in source.receiver_list:
                # wrt f, need possibility wrt m
                PTv = rx.getP(self.mesh, rx.projGLoc(f)).toarray().T

                df_duTFun = getattr(f, '_{0!s}Deriv'.format(rx.projField),
                                    None)
                df_duT, df_dmT = df_duTFun(source, None, PTv, adjoint=True)

                # Compute block of receivers
                ATinvdf_duT = self.Ainv * df_duT

                if len(ATinvdf_duT.shape) == 1:
                    ATinvdf_duT = np.c_[ATinvdf_duT]

                dA_dmT = self.getADeriv(u_source, ATinvdf_duT, adjoint=True)

                dRHS_dmT = self.getRHSDeriv(source, ATinvdf_duT, adjoint=True)

                du_dmT = -da.from_delayed(dA_dmT, shape=(self.model.size, rx.nD), dtype=float) + da.from_delayed(dRHS_dmT, shape=(self.model.size, rx.nD), dtype=float)

                if not isinstance(df_dmT, Zero):

                    du_dmT += da.from_delayed(df_dmT, shape=(self.model.size, rx.nD), dtype=float)

                blockName = self.Jpath + "J" + str(count) + ".zarr"
                nChunks = self.n_cpu  # Number of chunks
                rowChunk = int(np.ceil(rx.nD/nChunks))
                colChunk = int(np.ceil(self.model.size/nChunks))  # Chunk sizes
                du_dmT = du_dmT.rechunk((colChunk, rowChunk))

                da.to_zarr(du_dmT, blockName)

                Jtv.append(du_dmT)
                count += 1


        # Stack all the source blocks in one big zarr
        J = da.hstack(Jtv).T
        nChunks = self.n_cpu  # Number of chunks
        rowChunk = int(np.ceil(self.survey.nD/nChunks))# Chunk sizes
        colChunk = int(np.ceil(m.shape[0]/nChunks))
        J = J.rechunk((rowChunk, colChunk))

        da.to_zarr(J, self.Jpath + "J.zarr")
        self._Jmatrix = da.from_zarr(self.Jpath + "J.zarr")
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
            return mkvc(da.dot(J, v).compute())

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

            return mkvc(da.dot(J.T, v).compute())

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

    @dask.delayed
    def getADeriv(self, u, v, adjoint=False):

        D = self.Div
        G = self.Grad
        MfRhoIDeriv = self.MfRhoIDeriv

        if adjoint:
            return MfRhoIDeriv(G * u, D.T * v, adjoint)

        return D * (MfRhoIDeriv(G * u, v, adjoint))

    def getRHS(self):
        """
        RHS for the DC problem
        q
        """

        RHS = self.getSourceTerm()

        return RHS

    @dask.delayed
    def getRHSDeriv(self, source, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = source.evalDeriv(self, adjoint=adjoint)
        # return qDeriv
        return Zero()

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

    @dask.delayed
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

    @dask.delayed
    def getRHSDeriv(self, source, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = source.evalDeriv(self, adjoint=adjoint)
        # return qDeriv
        return Zero()
