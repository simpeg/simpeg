import numpy as np
import scipy as sp
import properties

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


class BaseDCSimulation(BaseEMSimulation):
    """
    Base DC Problem
    """
    Jpath = "./sensitivity.zarr"
    n_cpu = None
    maxRAM = 2

    survey = properties.Instance(
        "a DC survey object", Survey, required=True
    )

    storeJ = properties.Bool(
        "store the sensitivity matrix?", default=False
    )

    Ainv = None
    _Jmatrix = None

    def testD(self):
        test_list = []
        for i in range(15):
            test_list.append(self.calculateTest())
        print(test_list[5])

    @dask.delayed
    def calculateTest(self):
        A = self.getA()
        self.Ainv = self.Solver(A, **self.solver_opts)
        RHS = self.getRHS()
        return self.Ainv * RHS

    def fields2(self, m=None):
        if m is not None:
            self.model = m

        if self.Ainv is not None:
            self.Ainv.clean()

        f = self.fieldsPair(self)
        A = self.getA()
        self.Ainv = dask.delayed(self.Solver)(A, **self.solver_opts)
        RHS = self.getRHS()
        u = self.Ainv * RHS
        Srcs = self.survey.srcList
        f[Srcs, self._solutionType] = u.compute()
        return f

    def fields(self, m=None):
        if m is not None:
            self.model = m

        if self.Ainv is not None:
            self.Ainv.clean()

        f = self.fieldsPair(self)
        A = self.getA().compute()
        self.Ainv = self.Solver(A, **self.solver_opts)
        RHS = self.getRHS()
        u = self.Ainv * RHS.compute()
        Srcs = self.survey.srcList
        f[Srcs, self._solutionType] = u
        return f

    def getJ(self, m, f=None):
        """
            Generate Full sensitivity matrix
        """
        if self.verbose:
            print("Calculating J and storing")

        if self._Jmatrix is not None:
            return self._Jmatrix
        else:

            self.model = m
            if f is None:
                f = self.fields(m)
            self._Jmatrix = (self._Jtvec(m, v=None, f=f)).T
        return self._Jmatrix

    def getJ2(self, m, f=None):
        """
            Generate Full sensitivity matrix
        """
        self.n_cpu = int(multiprocessing.cpu_count())
        if self.verbose:
            print("Calculating J and storing")

        if self._Jmatrix is not None:
            return self._Jmatrix
        else:

            self.model = m
            if f is None:
                f = self.fields2(m)
            J = (self._Jtvec2(m, v=None, f=f)).T

            nChunks = self.n_cpu  # Number of chunks
            nDataComps = 1
            rowChunk, colChunk = int(np.ceil(self.survey.nD*nDataComps/nChunks)), int(np.ceil(self.model.size/nChunks))  # Chunk sizes
            # J.rechunk((rowChunk, colChunk))
            print('DASK: ')
            print('Tile size (nD, nC): ', J.shape)
            print('Chunk sizes (nD, nC): ', rowChunk, colChunk) # For debugging only
            print('Number of chunks: ', len(J.chunks[0]), ' x ', len(J.chunks[1]), ' = ', len(J.chunks[0]) * len(J.chunks[1]))
            print("Target chunk size: ", dask.config.get('array.chunk-size'))
            print('Max chunk size (GB): ', max(J.chunks[0]) * max(J.chunks[1]) * 8 * 1e-9)
            print('Max RAM (GB x CPU): ', max(J.chunks[0]) * max(J.chunks[1]) * 8 * 1e-9 * self.n_cpu)
            print('Tile size (GB): ', J.shape[0] * J.shape[1] * 8 * 1e-9)
            print("Saving G to zarr: " + self.Jpath)
            da.to_zarr(J, self.Jpath)
            self._Jmatrix = da.from_zarr(self.Jpath)

        return self._Jmatrix

    def Jvec(self, m, v, f=None):
        """
            Compute sensitivity matrix (J) and vector (v) product.
        """
        if self.storeJ:
            J = self.getJ(m, f=f)
            Jv = mkvc(np.dot(J, v))
            return Jv

        self.model = m

        if f is None:
            f = self.fields(m)

        Jv = []

        for src in self.survey.srcList:
            u_src = f[src, self._solutionType]  # solution vector
            dA_dm_v = self.getADeriv(u_src, v)
            dRHS_dm_v = self.getRHSDeriv(src, v)
            du_dm_v = self.Ainv * (- dA_dm_v + dRHS_dm_v)
            for rx in src.rxList:
                df_dmFun = getattr(f, '_{0!s}Deriv'.format(rx.projField), None)
                df_dm_v = df_dmFun(src, du_dm_v, v, adjoint=False)
                Jv.append(rx.evalDeriv(src, self.mesh, f, df_dm_v))
        return np.hstack(Jv)

    def Jtvec(self, m, v, f=None):
        """
            Compute adjoint sensitivity matrix (J^T) and vector (v) product.

        """
        if self.storeJ:
            J = self.getJ(m, f=f)
            Jtv = mkvc(np.dot(J.T, v))
            return Jtv

        self.model = m

        if f is None:
            f = self.fields(m)

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

        for src in self.survey.srcList:
            u_src = f[src, self._solutionType].copy()
            for rx in src.rxList:
                # wrt f, need possibility wrt m
                if v is not None:
                    PTv = rx.evalDeriv(
                        src, self.mesh, f, v[src, rx], adjoint=True
                    )
                else:
                    # This is for forming full sensitivity matrix
                    PTv = rx.getP(self.mesh, rx.projGLoc(f)).toarray().T
                df_duTFun = getattr(f, '_{0!s}Deriv'.format(rx.projField),
                                    None)
                df_duT, df_dmT = df_duTFun(src, None, PTv, adjoint=True)

                ATinvdf_duT = self.Ainv * df_duT

                dA_dmT = self.getADeriv(u_src, ATinvdf_duT, adjoint=True).compute()
                dRHS_dmT = self.getRHSDeriv(src, ATinvdf_duT, adjoint=True).compute()
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

    def _Jtvec2(self, m, v=None, f=None):
        """
            Compute adjoint sensitivity matrix (J^T) and vector (v) product.
            Full J matrix can be computed by inputing v=None
        """
        if v is not None:
            # Ensure v is a data object.
            if not isinstance(v, Data):
                v = Data(self.survey, v)
            # Jtv = np.zeros(m.size)
            Jtv = []
        else:
            # This is for forming full sensitivity matrix
            # Jtv2 = da.zeros((self.model.size, self.survey.nD), order='F')
            Jtv = []
        for src in self.survey.srcList:
            u_src = f[src, self._solutionType].copy()
            for rx in src.rxList:
                # wrt f, need possibility wrt m
                if v is not None:
                    PTv = rx.evalDeriv(
                        src, self.mesh, f, v[src, rx], adjoint=True
                    )
                else:
                    # This is for forming full sensitivity matrix
                    # PTv = dask.delayed(rx.getP)(self.mesh, rx.projGLoc(f), transpose=True)
                    PTv = rx.getP(self.mesh, rx.projGLoc(f)).toarray().T

                # NEED TO CHECK WHAT DERIV TO BE USED LIKE PREVIOUS IMPLIMENTATION, NOT ALWAYS PHIDERIV
                df_duT = dask.delayed(f._phiDeriv_u)(src, PTv, adjoint=True)
                df_dmT = dask.delayed(f._phiDeriv_m)(src, PTv, adjoint=True)

                ATinvdf_duT = self.Ainv * df_duT
                dA_dmT = self.getADeriv(u_src, ATinvdf_duT, adjoint=True)
                dRHS_dmT = self.getRHSDeriv(src, ATinvdf_duT, adjoint=True)
                du_dmT = -dA_dmT + dRHS_dmT
                # type(df_dmT + du_dmT)
                if v is not None:
                    Jtv.append(da.from_delayed(df_dmT + du_dmT, (self.model.size,), dtype=float))
                    # Jtv += (df_dmT.compute() + du_dmT..astype(float)
                else:
                    Jtv.append(da.from_delayed(df_dmT + du_dmT, (self.model.size, rx.nD), dtype=float))

        if v is not None:
            # Jtv_ = da.sum(da.hstack(Jtv), axis=0)
            return da.sum(da.hstack(Jtv), axis=0)
        else:
            # return da.hstack(Jtv)
            return da.hstack(Jtv)

    def getSourceTerm(self):
        """
        Evaluates the sources, and puts them in matrix form
        :rtype: tuple
        :return: q (nC or nN, nSrc)
        """

        Srcs = self.survey.srcList

        if self._formulation == 'EB':
            n = self.mesh.nN
            # return NotImplementedError

        elif self._formulation == 'HJ':
            n = self.mesh.nC

        q = np.zeros((n, len(Srcs)))

        for i, src in enumerate(Srcs):
            q[:, i] = src.eval(self)
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
    bc_type = 'Neumann'

    def __init__(self, mesh, **kwargs):

        BaseDCSimulation.__init__(self, mesh, **kwargs)
        self.setBC()

    @dask.delayed
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

    @dask.delayed
    def getRHS(self):
        """
        RHS for the DC problem
        q
        """

        RHS = self.getSourceTerm()

        return RHS

    @dask.delayed
    def getRHSDeriv(self, src, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = src.evalDeriv(self, adjoint=adjoint)
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

    def getRHSDeriv(self, src, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = src.evalDeriv(self, adjoint=adjoint)
        # return qDeriv
        return Zero()
