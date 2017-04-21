from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from SimPEG import Utils
from SimPEG.EM.Base import BaseEMProblem
from SimPEG.EM.Static.DC.FieldsDC_2D import (
    Fields_ky, Fields_ky_CC, Fields_ky_N
    )
from SimPEG.EM.Static.DC import BaseDCProblem_2D
import numpy as np
from SimPEG.Utils import Zero
from SimPEG.EM.Static.DC import getxBCyBC_CC
from .SurveyIP import Survey
from SimPEG import Props


class BaseIPProblem_2D(BaseDCProblem_2D):

    sigma = Props.PhysicalProperty(
        "Electrical conductivity (S/m)"
    )

    rho = Props.PhysicalProperty(
        "Electrical resistivity (Ohm m)"
    )

    Props.Reciprocal(sigma, rho)

    eta, etaMap, etaDeriv = Props.Invertible(
        "Electrical Chargeability (V/V)"
    )

    surveyPair = Survey
    fieldsPair = Fields_ky
    J = None
    fswitch = False
    sign = None
    f = None

    def fieldsdc(self, m):
        if self.verbose:
            print (">> Compute DC fields")
        if m is not None:
            self.model = m
        if self.Ainv[0] is not None:
            for i in range(self.nky):
                self.Ainv[i].clean()

        self.f = self.fieldsPair(self.mesh, self.survey)
        Srcs = self.survey.srcList
        for iky in range(self.nky):
            ky = self.kys[iky]
            A = self.getA(ky)
            self.Ainv[iky] = self.Solver(A, **self.solverOpts)
            RHS = self.getRHS(ky)
            u = self.Ainv[iky] * RHS
            self.f[Srcs, self._solutionType, iky] = u

    def fields(self, m):
        # This is stupid..., but not sure how I can change InvProblem
        # where calling self.fields
        return None

    def getJ(self, m, f=None):
        """
            Generate Full sensitivity matrix
        """

        if self.verbose:
            print (">> Compute Sensitivity matrix")

        if self.f is None:
            self.fieldsdc(m)
        self.model = m

        Jt = []

        # Assume y=0.
        # This needs some thoughts to implement in general when src is dipole
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
                    u_src = self.f[src, self._solutionType, iky]
                    ky = self.kys[iky]
                    AT = self.getA(ky)

                    # wrt f, need possibility wrt m
                    P = rx.getP(self.mesh, rx.projGLoc(self.f)).toarray()

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

        return np.hstack(Jt).T

    def Jvec(self, m, v, f=None):
        self.model = m
        if self.fswitch == False:
            f = self.fieldsdc(m)
            self.J = self.getJ(m, f=f)
            self.fswitch = True
        return self.sign * self.J.dot(v)

    def Jtvec(self, m, v, f=None):
        self.model = m
        if self.fswitch == False:
            f = self.fieldsdc(m)
            self.J = self.getJ(m, f=f)
            self.fswitch = True
        Jtvec = self.J.T.dot(v)
        return self.sign * Jtvec

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
        drho_dlogrho = Utils.sdiag(self.rho)*self.etaDeriv
        return dMfRhoI_dI * (dMf_drho * drho_dlogrho)

    # TODO: This should take a vector
    def MeSigmaDeriv(self, u):
        """
            Derivative of MeSigma with respect to the model
        """
        dsigma_dlogsigma = Utils.sdiag(self.sigma)*self.etaDeriv
        return self.mesh.getEdgeInnerProductDeriv(self.sigma)(u) * dsigma_dlogsigma

    # assume log rho or log cond
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
        vol = self.mesh.vol
        dsigma_dlogsigma = Utils.sdiag(self.sigma)*self.etaDeriv
        return (
            Utils.sdiag(u)*self.mesh.aveN2CC.T *
            (Utils.sdiag(vol) * dsigma_dlogsigma)
                )

    @property
    def MccRhoI(self):
        """
            Cell inner product matrix for \\(\\sigma\\). Used in the H-J
            formulation
        """
        # TODO: only works isotropic sigma
        rho = self.rho
        vol = self.mesh.vol
        MccRhoI = Utils.sdiag(1./(Utils.sdiag(vol)*rho))
        return MccRhoI

    def MccRhoIDeriv(self, u):
        """
            Derivative of MccRhoI with respect to the model
        """
        rho = self.rho
        vol = self.mesh.vol
        drho_dlogrho = Utils.sdiag(rho)*self.etaDeriv
        return (
            Utils.sdiag(u.flatten()*vol*(-1./rho**2))*drho_dlogrho
            )


class Problem2D_CC(BaseIPProblem_2D):
    """
    2.5D cell centered IP problem
    """

    _solutionType = 'phiSolution'
    _formulation = 'HJ'  # CC potentials means J is on faces
    fieldsPair = Fields_ky_CC
    sign = 1.

    def __init__(self, mesh, **kwargs):
        BaseIPProblem_2D.__init__(self, mesh, **kwargs)
        self.setBC()

    def getA(self, ky):
        """

        Make the A matrix for the cell centered DC resistivity problem

        A = D MfRhoI G

        """

        D = self.Div
        G = self.Grad
        vol = self.mesh.vol
        MfRhoI = self.MfRhoI
        # Get resistivity rho
        rho = self.rho
        A = D * MfRhoI * G + Utils.sdiag(ky**2*vol/rho)
        return A

    def getADeriv(self, ky, u, v, adjoint=False):

        D = self.Div
        G = self.Grad
        vol = self.mesh.vol
        MfRhoIDeriv = self.MfRhoIDeriv
        MccRhoIDeriv = self.MccRhoIDeriv
        rho = self.rho
        if adjoint:
            return((MfRhoIDeriv( G * u).T) * (D.T * v) +
                   ky**2 * MccRhoIDeriv(u).T * v)
        return (D * ((MfRhoIDeriv(G * u)) * v) + ky**2*MccRhoIDeriv(u)*v)

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
            temp_xm = np.ones_like(gBFxm[:, 0])
            temp_xp = np.ones_like(gBFxp[:, 0])
            temp_ym = np.ones_like(gBFym[:, 1])
            temp_yp = np.ones_like(gBFyp[:, 1])
            temp_zm = np.ones_like(gBFzm[:, 2])
            temp_zp = np.ones_like(gBFzp[:, 2])

            alpha_xm, alpha_xp = temp_xm*0., temp_xp*0.
            alpha_ym, alpha_yp = temp_ym*0., temp_yp*0.
            alpha_zm, alpha_zp = temp_zm*0., temp_zp*0.

            beta_xm, beta_xp = temp_xm, temp_xp
            beta_ym, beta_yp = temp_ym, temp_yp
            beta_zm, beta_zp = temp_zm, temp_zp

            gamma_xm, gamma_xp = temp_xm*0., temp_xp*0.
            gamma_ym, gamma_yp = temp_ym*0., temp_yp*0.
            gamma_zm, gamma_zp = temp_zm*0., temp_zp*0.

            alpha = [alpha_xm, alpha_xp, alpha_ym, alpha_yp, alpha_zm,
                     alpha_zp]
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
        self.Grad = self.Div.T - P_BC*Utils.sdiag(y_BC)*M


class Problem2D_N(BaseIPProblem_2D):
    """
    2.5D nodal IP problem
    """

    _solutionType = 'phiSolution'
    _formulation = 'EB'  # CC potentials means J is on faces
    fieldsPair = Fields_ky_N
    sign = -1.

    def __init__(self, mesh, **kwargs):
        BaseIPProblem_2D.__init__(self, mesh, **kwargs)
        # self.setBC()

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

        # Handling Null space of A
        A[0, 0] = A[0, 0] + 1.
        return A

    @Utils.count
    @Utils.timeIt
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
