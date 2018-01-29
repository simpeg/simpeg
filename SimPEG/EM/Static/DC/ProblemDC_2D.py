from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from SimPEG import Utils
from SimPEG.EM.Base import BaseEMProblem
from .SurveyDC import Survey_ky
from .FieldsDC_2D import Fields_ky, Fields_ky_CC, Fields_ky_N
from .FieldsDC import FieldsDC, Fields_CC, Fields_N
import numpy as np
from SimPEG.Utils import Zero
from .BoundaryUtils import getxBCyBC_CC
from scipy.special import kn


class BaseDCProblem_2D(BaseEMProblem):
    """
    Base 2.5D DC problem
    """
    surveyPair = Survey_ky
    fieldsPair = Fields_ky  # SimPEG.EM.Static.Fields_2D
    fieldsPair_fwd = FieldsDC
    nky = 15
    kys = np.logspace(-4, 1, nky)
    Ainv = [None for i in range(nky)]
    nT = nky  # Only for using TimeFields
    storeJ = False
    _Jmatrix = None

    def fields(self, m):
        if m is not None:
            self.model = m
        if self.Ainv[0] is not None:
            for i in range(self.nky):
                self.Ainv[i].clean()

        f = self.fieldsPair(self.mesh, self.survey)
        Srcs = self.survey.srcList
        for iky in range(self.nky):
            ky = self.kys[iky]
            A = self.getA(ky)
            self.Ainv[iky] = self.Solver(A, **self.solverOpts)
            RHS = self.getRHS(ky)
            u = self.Ainv[iky] * RHS
            f[Srcs, self._solutionType, iky] = u
        return f

    def fields_to_space(self, f, y=0.):
        f_fwd = self.fieldsPair_fwd(self.mesh, self.survey)
        # Evaluating Integration using Trapezoidal rules
        nky = self.kys.size
        dky = np.diff(self.kys)
        dky = np.r_[dky[0], dky]
        phi0 = 1./np.pi*f[:, self._solutionType, 0]
        phi = np.zeros_like(phi0)
        for iky in range(nky):
            phi1 = 1./np.pi*f[:, self._solutionType, iky]
            phi += phi1*dky[iky]/2.*np.cos(self.kys[iky]*y)
            phi += phi0*dky[iky]/2.*np.cos(self.kys[iky]*y)
            phi0 = phi1.copy()
        f_fwd[:, self._solutionType] = phi
        return f_fwd

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

    def Jvec(self, m, v, f=None):
        """
            Compute sensitivity matrix (J) and vector (v) product.
        """
        if self.storeJ:
            J = self.getJ(m, f=f)
            Jv = Utils.mkvc(np.dot(J, v))
            return Jv

        self.model = m

        if f is None:
            f = self.fields(m)

        # TODO: This is not a good idea !! should change that as a list
        Jv = self.dataPair(self.survey)  # same size as the data
        Jv0 = self.dataPair(self.survey)

        # Assume y=0.
        # This needs some thoughts to implement in general when src is dipole
        dky = np.diff(self.kys)
        dky = np.r_[dky[0], dky]
        y = 0.

        # TODO: this loop is pretty slow .. (Parellize)
        for iky in range(self.nky):
            ky = self.kys[iky]
            A = self.getA(ky)
            for src in self.survey.srcList:
                u_src = f[src, self._solutionType, iky]  # solution vector
                dA_dm_v = self.getADeriv(ky, u_src, v)
                dRHS_dm_v = self.getRHSDeriv(ky, src, v)
                du_dm_v = self.Ainv[iky] * (- dA_dm_v + dRHS_dm_v)
                for rx in src.rxList:
                    df_dmFun = getattr(f, '_{0!s}Deriv'.format(rx.projField),
                                       None)
                    df_dm_v = df_dmFun(iky, src, du_dm_v, v, adjoint=False)
                    # Trapezoidal intergration
                    Jv1_temp = 1./np.pi*rx.evalDeriv(ky, src, self.mesh, f,
                                                     df_dm_v)
                    if iky == 0:
                        # First assigment
                        Jv[src, rx] = Jv1_temp*dky[iky]*np.cos(ky*y)
                    else:
                        Jv[src, rx] += Jv1_temp*dky[iky]/2.*np.cos(ky*y)
                        Jv[src, rx] += Jv0[src, rx]*dky[iky]/2.*np.cos(ky*y)
                    Jv0[src, rx] = Jv1_temp.copy()
        return Utils.mkvc(Jv)

    def Jtvec(self, m, v, f=None):
        """
            Compute adjoint sensitivity matrix (J^T) and vector (v) product.

        """
        if self.storeJ:
            J = self.getJ(m, f=f)
            Jtv = Utils.mkvc(np.dot(J.T, v))
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
            if not isinstance(v, self.dataPair):
                v = self.dataPair(self.survey, v)
            Jtv = np.zeros(m.size, dtype=float)

            # Assume y=0.
            dky = np.diff(self.kys)
            dky = np.r_[dky[0], dky]
            y = 0.

            for src in self.survey.srcList:
                for rx in src.rxList:
                    Jtv_temp1 = np.zeros(m.size, dtype=float)
                    Jtv_temp0 = np.zeros(m.size, dtype=float)

                    # TODO: this loop is pretty slow .. (Parellize)
                    for iky in range(self.nky):
                        u_src = f[src, self._solutionType, iky]
                        ky = self.kys[iky]
                        AT = self.getA(ky)
                        # wrt f, need possibility wrt m
                        PTv = rx.evalDeriv(ky, src, self.mesh, f, v[src, rx],
                                           adjoint=True)
                        df_duTFun = getattr(
                            f, '_{0!s}Deriv'.format(rx.projField), None
                        )
                        df_duT, df_dmT = df_duTFun(iky, src, None, PTv,
                                                   adjoint=True)

                        ATinvdf_duT = self.Ainv[iky] * df_duT

                        dA_dmT = self.getADeriv(ky, u_src, ATinvdf_duT,
                                                adjoint=True)
                        dRHS_dmT = self.getRHSDeriv(ky, src, ATinvdf_duT,
                                                    adjoint=True)
                        du_dmT = -dA_dmT + dRHS_dmT
                        Jtv_temp1 = 1./np.pi*(df_dmT + du_dmT).astype(float)
                        # Trapezoidal intergration
                        if iky == 0:
                            # First assigment
                            Jtv += Jtv_temp1*dky[iky]*np.cos(ky*y)
                        else:
                            Jtv += Jtv_temp1*dky[iky]/2.*np.cos(ky*y)
                            Jtv += Jtv_temp0*dky[iky]/2.*np.cos(ky*y)
                        Jtv_temp0 = Jtv_temp1.copy()
            return Utils.mkvc(Jtv)

        # This is for forming full sensitivity
        else:

            Jt = []

            # Assume y=0.
            dky = np.diff(self.kys)
            dky = np.r_[dky[0], dky]
            y = 0.
            for src in self.survey.srcList:
                for rx in src.rxList:
                    Jtv_temp1 = np.zeros((m.size, rx.nD), dtype=float)
                    Jtv_temp0 = np.zeros((m.size, rx.nD), dtype=float)
                    Jtv = np.zeros((m.size, rx.nD), dtype=float)
                    # TODO: this loop is pretty slow .. (Parellize)
                    for iky in range(self.nky):
                        u_src = f[src, self._solutionType, iky]
                        ky = self.kys[iky]
                        AT = self.getA(ky)

                        # wrt f, need possibility wrt m
                        P = rx.getP(self.mesh, rx.projGLoc(f)).toarray()

                        ATinvdf_duT = self.Ainv[iky] * (P.T)

                        dA_dmT = self.getADeriv(ky, u_src, ATinvdf_duT,
                                                adjoint=True)
                        Jtv_temp1 = 1./np.pi*(-dA_dmT)
                        if rx.nD == 1:
                            Jtv_temp1 = Jtv_temp1.reshape([-1, 1])

                        # Trapezoidal intergration
                        if iky == 0:
                            # First assigment
                            Jtv += Jtv_temp1*dky[iky]*np.cos(ky*y)
                        else:
                            Jtv += Jtv_temp1*dky[iky]/2.*np.cos(ky*y)
                            Jtv += Jtv_temp0*dky[iky]/2.*np.cos(ky*y)
                        Jtv_temp0 = Jtv_temp1.copy()

                    Jt.append(Jtv)
            return np.hstack(Jt)

    def getSourceTerm(self, ky):
        """
        takes concept of source and turns it into a matrix
        """
        """
        Evaluates the sources, and puts them in matrix form

        :rtype: (numpy.ndarray, numpy.ndarray)
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
        toDelete = []
        if self.sigmaMap is not None or self.rhoMap is not None:
            toDelete += ['_MeSigma', '_MeSigmaI', '_MfRho', '_MfRhoI']
        if self._Jmatrix is not None:
            toDelete += ['_Jmatrix']
        return toDelete


class Problem2D_CC(BaseDCProblem_2D):
    """
    2.5D cell centered DC problem
    """

    _solutionType = 'phiSolution'
    _formulation = 'HJ'  # CC potentials means J is on faces
    fieldsPair = Fields_ky_CC
    fieldsPair_fwd = Fields_CC
    bc_type = 'Mixed'

    def __init__(self, mesh, **kwargs):
        BaseDCProblem_2D.__init__(self, mesh, **kwargs)

    def getA(self, ky):
        """

        Make the A matrix for the cell centered DC resistivity problem

        A = D MfRhoI G

        """
        # To handle Mixed boundary condition
        if self._formulation == "HJ":
            self.setBC(ky=ky)

        D = self.Div
        G = self.Grad
        vol = self.mesh.vol
        MfRhoI = self.MfRhoI
        # Get resistivity rho
        rho = self.rho
        A = D * MfRhoI * G + Utils.sdiag(ky**2*vol/rho)
        if self.bc_type == "Neumann":
            A[0, 0] = A[0, 0] + 1.
        return A

    def getADeriv(self, ky, u, v, adjoint=False):

        # To handle Mixed boundary condition
        if self._formulation == "HJ":
            self.setBC(ky=ky)

        D = self.Div
        G = self.Grad
        vol = self.mesh.vol
        MfRhoIDeriv = self.MfRhoIDeriv
        rho = self.rho
        if adjoint:
            return((MfRhoIDeriv( G * u).T) * (D.T * v) +
                   ky**2 * self.rhoDeriv.T*Utils.sdiag(u.flatten()*vol*(-1./rho**2))*v)

        return (D * ((MfRhoIDeriv(G * u)) * v) + ky**2*
                Utils.sdiag(u.flatten()*vol*(-1./rho**2))*(self.rhoDeriv*v))

    def getRHS(self, ky):
        """
        RHS for the DC problem

        q
        """

        RHS = self.getSourceTerm(ky)
        return RHS

    def getRHSDeriv(self, ky, src, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = src.evalDeriv(self, ky, adjoint=adjoint)
        # return qDeriv
        return Zero()

    def setBC(self, ky=None):
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

        if self.bc_type == "Neumann":
            alpha_xm, alpha_xp = temp_xm*0., temp_xp*0.
            alpha_ym, alpha_yp = temp_ym*0., temp_yp*0.

            beta_xm, beta_xp = temp_xm, temp_xp
            beta_ym, beta_yp = temp_ym, temp_yp

            gamma_xm, gamma_xp = temp_xm*0., temp_xp*0.
            gamma_ym, gamma_yp = temp_ym*0., temp_yp*0.

        elif self.bc_type == "Dirichlet":
            alpha_xm, alpha_xp = temp_xm, temp_xp
            alpha_ym, alpha_yp = temp_ym, temp_yp

            beta_xm, beta_xp = temp_xm*0., temp_xp*0.
            beta_ym, beta_yp = temp_ym*0., temp_yp*0.

            gamma_xm, gamma_xp = temp_xm*0., temp_xp*0.
            gamma_ym, gamma_yp = temp_ym*0., temp_yp*0.

        elif self.bc_type == "Mixed":
            xs = np.median(self.mesh.vectorCCx)
            ys = np.median(self.mesh.vectorCCy[-1])

            def r_boundary(x, y):
                return 1./np.sqrt(
                    (x - xs)**2 + (y - ys)**2
                    )

            rxm = r_boundary(gBFxm[:, 0], gBFxm[:, 1])
            rxp = r_boundary(gBFxp[:, 0], gBFxp[:, 1])
            rym = r_boundary(gBFym[:, 0], gBFym[:, 1])

            alpha_xm = ky*(
                kn(1, ky*rxm) / kn(0, ky*rxm) * (gBFxm[:, 0]-xs)
                )
            alpha_xp = ky*(
                kn(1, ky*rxp) / kn(0, ky*rxp) * (gBFxp[:, 0]-xs)
                )
            alpha_ym = ky*(
                kn(1, ky*rym) / kn(0, ky*rym) * (gBFym[:, 0]-ys)
                )
            alpha_yp = temp_yp*0.
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
        self.Grad = self.Div.T - P_BC*Utils.sdiag(y_BC)*M


class Problem2D_N(BaseDCProblem_2D):
    """
    2.5D nodal DC problem
    """

    _solutionType = 'phiSolution'
    _formulation = 'EB'  # CC potentials means J is on faces
    fieldsPair = Fields_ky_N
    fieldsPair_fwd = Fields_N

    def __init__(self, mesh, **kwargs):
        BaseDCProblem_2D.__init__(self, mesh, **kwargs)
        # self.setBC()

    @property
    def MnSigma(self):
        """
            Node inner product matrix for \\(\\sigma\\). Used in the E-B
            formulation
        """
        # TODO: only works isotropic sigma
        sigma = self.sigma
        vol = self.mesh.vol
        MnSigma = Utils.sdiag(self.mesh.aveN2CC.T*(Utils.sdiag(vol)*sigma))

        return MnSigma

    def MnSigmaDeriv(self, u):
        """
            Derivative of MnSigma with respect to the model
        """
        sigma = self.sigma
        sigmaderiv = self.sigmaDeriv
        vol = self.mesh.vol
        return (Utils.sdiag(u)*self.mesh.aveN2CC.T*Utils.sdiag(vol) *
                self.sigmaDeriv)

    def getA(self, ky):
        """

        Make the A matrix for the cell centered DC resistivity problem

        A = D MfRhoI G

        """

        MeSigma = self.MeSigma
        MnSigma = self.MnSigma
        Grad = self.mesh.nodalGrad
        # Get conductivity sigma
        sigma = self.sigma
        A = Grad.T * MeSigma * Grad + ky**2*MnSigma
        # This seems not required for 2.5D problem
        # Handling Null space of A
        # A[0, 0] = A[0, 0] + 1.
        return A

    def getADeriv(self, ky, u, v, adjoint=False):

        MeSigma = self.MeSigma
        Grad = self.mesh.nodalGrad
        sigma = self.sigma
        vol = self.mesh.vol

        if adjoint:
            return (self.MeSigmaDeriv(Grad*u).T * (Grad*v) +
                    ky**2*self.MnSigmaDeriv(u).T*v)
        return (Grad.T*(self.MeSigmaDeriv(Grad*u)*v) +
                ky**2*self.MnSigmaDeriv(u)*v)

    def getRHS(self, ky):
        """
        RHS for the DC problem

        q
        """

        RHS = self.getSourceTerm(ky)
        return RHS

    def getRHSDeriv(self, ky, src, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = src.evalDeriv(self, ky, adjoint=adjoint)
        # return qDeriv
        return Zero()
