from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from SimPEG import Utils
from SimPEG import Props

from SimPEG.EM.Base import BaseEMProblem
from SimPEG.EM.Static.DC.FieldsDC import FieldsDC, Fields_CC, Fields_N
from SimPEG.EM.Static.DC import getxBCyBC_CC
from .SurveySIP import Survey, Data


class BaseSIPProblem(BaseEMProblem):

    eta, etaMap, etaDeriv = Props.Invertible(
        "Electrical Chargeability"
    )

    tau, tauMap, tauDeriv = Props.Invertible(
        "time constant",
        default=0.1
    )

    taui, tauiMap, tauiDeriv = Props.Invertible(
        "inverse time constant"
    )

    Props.Reciprocal(tau, taui)

    c, cMap, cDeriv = Props.Invertible(
        "frequency dependency",
        default=1.
    )

    surveyPair = Survey
    fieldsPair = FieldsDC
    dataPair = Data
    Ainv = None
    sigma = None
    rho = None
    f = None
    Ainv = None

    def DebyeTime(self, t):
        peta = self.eta*np.exp(-self.taui*t)
        return peta

    def EtaDeriv(self, t, v, adjoint=False):
        v = np.array(v, dtype=float)
        if adjoint:
            return self.etaDeriv.T * (np.exp(-self.taui*t)*v)
        else:
            return np.exp(-self.taui*t) * (self.etaDeriv*v)

    def TauiDeriv(self, t, v, adjoint=False):
        v = np.array(v, dtype=float)
        if adjoint:
            return -self.tauiDeriv.T * (self.eta*t*np.exp(-self.taui*t)*v)
        else:
            return -self.eta*t*np.exp(-self.taui*t) * (self.tauiDeriv*v)

    def fields(self, m):
        self.model = m
        if self.f is None:
            self.f = self.fieldsPair(self.mesh, self.survey)
            if self.Ainv is None:
                A = self.getA()
                self.Ainv = self.Solver(A, **self.solverOpts)
            RHS = self.getRHS()
            u = self.Ainv * RHS
            Srcs = self.survey.srcList
            self.f[Srcs, self._solutionType] = u
        return self.f

    def forward(self, m, f=None):

        if f is None:
            f = self.fields(m)

        self.model = m
        Jv = []

        # A = self.getA()
        for tind in range(len(self.survey.times)):
            # Pseudo-chareability
            t = self.survey.times[tind]
            v = self.DebyeTime(t)
            for src in self.survey.srcList:
                u_src = f[src, self._solutionType]  # solution vector
                dA_dm_v = self.getADeriv(u_src, v)
                dRHS_dm_v = self.getRHSDeriv(src, v)
                du_dm_v = self.Ainv * (- dA_dm_v + dRHS_dm_v)
                for rx in src.rxList:
                    timeindex = rx.getTimeP(self.survey.times)
                    if timeindex[tind]:
                        df_dmFun = getattr(f, '_{0!s}Deriv'.format(rx.projField), None)
                        df_dm_v = df_dmFun(src, du_dm_v, v, adjoint=False)
                        Jv.append(rx.evalDeriv(src, self.mesh, f, df_dm_v))
                        # Jv[src, rx, t] = rx.evalDeriv(src, self.mesh, f, df_dm_v)

        # Conductivity (d u / d log sigma)
        if self._formulation == 'EB':
            # return -Utils.mkvc(Jv)
            return -np.hstack(Jv)
        # Resistivity (d u / d log rho)
        if self._formulation == 'HJ':
            # return Utils.mkvc(Jv)
            return np.hstack(Jv)

    def Jvec(self, m, v, f=None):

        if f is None:
            f = self.fields(m)

        self.model = m
        Jv = []
        # A = self.getA()
        JvAll = []

        for tind in range(len(self.survey.times)):
            t = self.survey.times[tind]
            v0 = self.EtaDeriv(t, v)
            v1 = self.TauiDeriv(t, v)
            for src in self.survey.srcList:
                u_src = f[src, self._solutionType]  # solution vector
                dA_dm_v0 = self.getADeriv(u_src, v0)
                dRHS_dm_v0 = self.getRHSDeriv(src, v0)
                du_dm_v0 = self.Ainv * (- dA_dm_v0 + dRHS_dm_v0)
                dA_dm_v1 = self.getADeriv(u_src, v1)
                dRHS_dm_v1 = self.getRHSDeriv(src, v1)
                du_dm_v1 = self.Ainv * (- dA_dm_v1 + dRHS_dm_v1)
                for rx in src.rxList:
                    timeindex = rx.getTimeP(self.survey.times)
                    if timeindex[tind]:
                        df_dmFun = getattr(f, '_{0!s}Deriv'.format(rx.projField), None)
                        df_dm_v0 = df_dmFun(src, du_dm_v0, v0, adjoint=False)
                        df_dm_v1 = df_dmFun(src, du_dm_v1, v1, adjoint=False)
                        # Jv[src, rx, t] = rx.evalDeriv(src, self.mesh, f, df_dm_v0)
                        # Jv[src, rx, t] += rx.evalDeriv(src, self.mesh, f, df_dm_v1)
                        Jv.append(rx.evalDeriv(src, self.mesh, f, df_dm_v0) +
                                  rx.evalDeriv(src, self.mesh, f, df_dm_v1))

        # Conductivity (d u / d log sigma)
        if self._formulation == 'EB':
            # return -Jv.tovec()
            return -np.hstack(Jv)
        # Resistivity (d u / d log rho)
        if self._formulation == 'HJ':
            # return Jv.tovec()
            return np.hstack(Jv)

    def Jtvec(self, m, v, f=None):
        if f is None:
            f = self.fields(m)

        self.model = m

        # Ensure v is a data object.
        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        Jtv = np.zeros(m.size)

        for tind in range(len(self.survey.times)):
            t = self.survey.times[tind]
            for src in self.survey.srcList:
                u_src = f[src, self._solutionType]
                for rx in src.rxList:
                    timeindex = rx.getTimeP(self.survey.times)
                    if timeindex[tind]:
                        PTv = rx.evalDeriv(src, self.mesh, f, v[src, rx, t], adjoint=True)  # wrt f, need possibility wrt m
                        df_duTFun = getattr(f, '_{0!s}Deriv'.format(rx.projField), None)
                        df_duT, df_dmT = df_duTFun(src, None, PTv, adjoint=True)
                        ATinvdf_duT = self.Ainv * df_duT
                        dA_dmT = self.getADeriv(u_src, ATinvdf_duT, adjoint=True)
                        dRHS_dmT = self.getRHSDeriv(src, ATinvdf_duT, adjoint=True)
                        du_dmT = -dA_dmT + dRHS_dmT
                        Jtv += self.EtaDeriv(self.survey.times[tind], du_dmT, adjoint=True) + self.TauiDeriv(self.survey.times[tind], du_dmT, adjoint=True)

        # Conductivity ((d u / d log sigma).T)
        if self._formulation == 'EB':
            return -Jtv
        # Conductivity ((d u / d log rho).T)
        if self._formulation == 'HJ':
            return Jtv

    def getSourceTerm(self):
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
        return toDelete

    # assume log rho or log cond
    @property
    def MeSigma(self):
        """
            Edge inner product matrix for \\(\\sigma\\).
            Used in the E-B formulation
        """
        if getattr(self, '_MeSigma', None) is None:
            self._MeSigma = self.mesh.getEdgeInnerProduct(self.sigma)
        return self._MeSigma

    @property
    def MfRhoI(self):
        """
            Inverse of :code:`MfRho`
        """
        if getattr(self, '_MfRhoI', None) is None:
            self._MfRhoI = self.mesh.getFaceInnerProduct(self.rho, invMat=True)
        return self._MfRhoI

    def MfRhoIDeriv(self, u):
        """
            Derivative of :code:`MfRhoI` with respect to the model.
        """

        dMfRhoI_dI = -self.MfRhoI**2
        dMf_drho = self.mesh.getFaceInnerProductDeriv(self.rho)(u)
        drho_dlogrho = Utils.sdiag(self.rho)
        return dMfRhoI_dI * (dMf_drho * drho_dlogrho)

    # TODO: This should take a vector
    def MeSigmaDeriv(self, u):
        """
            Derivative of MeSigma with respect to the model
        """
        dsigma_dlogsigma = Utils.sdiag(self.sigma)
        return self.mesh.getEdgeInnerProductDeriv(self.sigma)(u) * dsigma_dlogsigma


class Problem3D_CC(BaseSIPProblem):

    _solutionType = 'phiSolution'
    _formulation = 'HJ'  # CC potentials means J is on faces
    fieldsPair = Fields_CC

    def __init__(self, mesh, **kwargs):
        BaseSIPProblem.__init__(self, mesh, **kwargs)
        self.setBC()

    def getA(self):
        """

        Make the A matrix for the cell centered DC resistivity problem

        A = D MfRhoI G

        """

        D = self.Div
        G = self.Grad
        # TODO: this won't work for full anisotropy
        MfRhoI = self.MfRhoI
        A = D * MfRhoI * G

        # I think we should deprecate this for DC problem.
        # if self._makeASymmetric is True:
        #     return V.T * A
        return A

    def getADeriv(self, u, v, adjoint=False):

        D = self.Div
        G = self.Grad
        MfRhoIDeriv = self.MfRhoIDeriv

        if adjoint:
            # if self._makeASymmetric is True:
            #     v = V * v
            return(MfRhoIDeriv(G * u).T) * (D.T * v)

        # I think we should deprecate this for DC problem.
        # if self._makeASymmetric is True:
        #     return V.T * ( D * ( MfRhoIDeriv( D.T * ( V * u ) ) * v ) )
        return D * (MfRhoIDeriv(G * u) * v)

    def getRHS(self):
        """
        RHS for the DC problem

        q
        """

        RHS = self.getSourceTerm()

        # I think we should deprecate this for DC problem.
        # if self._makeASymmetric is True:
        #     return self.Vol.T * RHS

        return RHS

    def getRHSDeriv(self, src, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = src.evalDeriv(self, adjoint=adjoint)
        # return qDeriv
        return Utils.Zero()

    def setBC(self):
        if self.mesh.dim == 3:
            fxm, fxp, fym, fyp, fzm, fzp = self.mesh.faceBoundaryInd
            gBFxm = self.mesh.gridFx[fxm, :]
            gBFxp = self.mesh.gridFx[fxp, :]
            gBFym = self.mesh.gridFy[fym, :]
            gBFyp = self.mesh.gridFy[fyp, :]
            gBFzm = self.mesh.gridFz[fzm, :]
            gBFzp = self.mesh.gridFz[fzp, :]

            # Setup Mixed B.C (alpha, beta, gamma)
            temp_xm, temp_xp = np.ones_like(gBFxm[:, 0]), np.ones_like(gBFxp[:, 0])
            temp_ym, temp_yp = np.ones_like(gBFym[:, 1]), np.ones_like(gBFyp[:, 1])
            temp_zm, temp_zp = np.ones_like(gBFzm[:, 2]), np.ones_like(gBFzp[:, 2])

            alpha_xm, alpha_xp = temp_xm*0., temp_xp*0.
            alpha_ym, alpha_yp = temp_ym*0., temp_yp*0.
            alpha_zm, alpha_zp = temp_zm*0., temp_zp*0.

            beta_xm, beta_xp = temp_xm, temp_xp
            beta_ym, beta_yp = temp_ym, temp_yp
            beta_zm, beta_zp = temp_zm, temp_zp

            gamma_xm, gamma_xp = temp_xm*0., temp_xp*0.
            gamma_ym, gamma_yp = temp_ym*0., temp_yp*0.
            gamma_zm, gamma_zp = temp_zm*0., temp_zp*0.

            alpha = [alpha_xm, alpha_xp, alpha_ym, alpha_yp, alpha_zm, alpha_zp]
            beta = [beta_xm, beta_xp, beta_ym, beta_yp, beta_zm, beta_zp]
            gamma = [gamma_xm, gamma_xp, gamma_ym, gamma_yp, gamma_zm, gamma_zp]

        elif self.mesh.dim == 2:

            fxm, fxp, fym, fyp = self.mesh.faceBoundaryInd
            gBFxm = self.mesh.gridFx[fxm, :]
            gBFxp = self.mesh.gridFx[fxp, :]
            gBFym = self.mesh.gridFy[fym, :]
            gBFyp = self.mesh.gridFy[fyp, :]

            # Setup Mixed B.C (alpha, beta, gamma)
            temp_xm, temp_xp = np.ones_like(gBFxm[:, 0]), np.ones_like(gBFxp[:, 0])
            temp_ym, temp_yp = np.ones_like(gBFym[:, 1]), np.ones_like(gBFyp[:, 1])

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
        self.Grad = self.Div.T - P_BC*Utils.sdiag(y_BC)*M


class Problem3D_N(BaseSIPProblem):

    _solutionType = 'phiSolution'
    _formulation = 'EB'  # N potentials means B is on faces
    fieldsPair = Fields_N

    def __init__(self, mesh, **kwargs):
        BaseSIPProblem.__init__(self, mesh, **kwargs)

    def getA(self):
        """
            Make the A matrix for the cell centered DC resistivity problem

            A = G.T MeSigma G
        """

        # TODO: this won't work for full anisotropy
        MeSigma = self.MeSigma
        Grad = self.mesh.nodalGrad
        A = Grad.T * MeSigma * Grad

        # Handling Null space of A
        A[0, 0] = A[0, 0] + 1.

        return A

    def getADeriv(self, u, v, adjoint=False):
        """
            Product of the derivative of our system matrix with respect to
            the model and a vector
        """
        Grad = self.mesh.nodalGrad
        if not adjoint:
            return Grad.T*(self.MeSigmaDeriv(Grad*u)*v)
        elif adjoint:
            return self.MeSigmaDeriv(Grad*u).T * (Grad*v)

    def getRHS(self):
        """
            RHS for the DC problem
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
        return Utils.Zero()
