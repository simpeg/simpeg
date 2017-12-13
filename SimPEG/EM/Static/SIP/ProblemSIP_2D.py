from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from SimPEG import Utils
from SimPEG import Props
from SimPEG import Maps

from SimPEG.EM.Static.DC.FieldsDC_2D import (
    Fields_ky, Fields_ky_CC, Fields_ky_N
    )
from SimPEG.EM.Static.DC import getxBCyBC_CC
from SimPEG.EM.Static.IP import BaseIPProblem_2D
from .SurveySIP import Survey


class BaseSIPProblem_2D(BaseIPProblem_2D):

    eta, etaMap, etaDeriv = Props.Invertible(
        "Electrical Chargeability (V/V)"
    )

    tau, tauMap, tauDeriv = Props.Invertible(
        "Time constant (s)",
        default=0.1
    )

    taui, tauiMap, tauiDeriv = Props.Invertible(
        "Inverse of time constant (1/s)"
    )

    Props.Reciprocal(tau, taui)

    c, cMap, cDeriv = Props.Invertible(
        "Frequency dependency",
        default=1.
    )

    surveyPair = Survey
    fieldsPair = Fields_ky
    f = None
    actinds = None
    actMap = None

    def getPeta(self, t):
        peta = self.eta*np.exp(-(self.taui*t)**self.c)
        return peta

    def PetaEtaDeriv(self, t, v, adjoint=False):
        # v = np.array(v, dtype=float)
        taui_t_c = (self.taui*t)**self.c
        dpetadeta = np.exp(-taui_t_c)
        if adjoint:
            return self.etaDeriv.T * (Utils.sdiag(dpetadeta) * v)
        else:
            return dpetadeta * (self.etaDeriv*v)

    def PetaTauiDeriv(self, t, v, adjoint=False):
        # v = np.array(v, dtype=float)
        taui_t_c = (self.taui*t)**self.c
        dpetadtaui = (
            - self.c * self.eta / self.taui * taui_t_c * np.exp(-taui_t_c)
            )
        if adjoint:
            return self.tauiDeriv.T * (Utils.sdiag(dpetadtaui)*v)
        else:
            return dpetadtaui * (self.tauiDeriv*v)

    def PetaCDeriv(self, t, v, adjoint=False):
        # v = np.array(v, dtype=float)
        taui_t_c = (self.taui*t)**self.c
        dpetadc = (
            -self.eta * (taui_t_c)*np.exp(-taui_t_c) * np.log(self.taui*t)
            )
        if adjoint:
            return self.cDeriv.T * (Utils.sdiag(dpetadc)*v)
        else:
            return dpetadc * (self.cDeriv*v)

    def getJ(self, f=None):
        """
            Generate Full sensitivity matrix
        """

        if self.verbose:
            print (">> Compute Sensitivity matrix")

        if self.f is None:
            self.fieldsdc()
        Jt = []

        # Assume y=0.
        # This needs some thoughts to implement in general when src is dipole
        dky = np.diff(self.kys)
        dky = np.r_[dky[0], dky]
        y = 0.
        for src in self.survey.srcList:
            for rx in src.rxList:
                Jtv_temp1 = np.zeros((self.actinds.sum(), rx.nD), dtype=float)
                Jtv_temp0 = np.zeros((self.actinds.sum(), rx.nD), dtype=float)
                Jtv = np.zeros((self.actinds.sum(), rx.nD), dtype=float)
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

    def forward(self, m, f=None):
        if self.fswitch == False:
            f = self.fieldsdc()
            self.J = self.getJ(f=f)
            self.fswitch = True

        ntime = len(self.survey.times)
        Jv = []
        self.model = m
        for tind in range(ntime):
            Jv.append(
                self.J.dot(
                    self.actMap.P.T*self.getPeta(self.survey.times[tind]))
                )
        return self.sign * np.hstack(Jv)

    def Jvec(self, m, v, f=None):

        self.model = m
        if self.fswitch == False:
            f = self.fieldsdc()
            self.J = self.getJ(f=f)
            self.fswitch = True

        ntime = len(self.survey.times)
        Jv = []

        for tind in range(ntime):

            t = self.survey.times[tind]
            v0 = self.PetaEtaDeriv(t, v)
            v1 = self.PetaTauiDeriv(t, v)
            v2 = self.PetaCDeriv(t, v)
            PTv = self.actMap.P.T*(v0+v1+v2)
            Jv.append(self.J.dot(PTv))

        return self.sign * np.hstack(Jv)

    def Jtvec(self, m, v, f=None):

        self.model = m
        if self.fswitch == False:
            f = self.fieldsdc()
            self.J = self.getJ(f=f)
            self.fswitch = True

        ntime = len(self.survey.times)
        Jtvec = np.zeros(m.size)
        v = v.reshape((int(self.survey.nD/ntime), ntime), order="F")

        for tind in range(ntime):
            t = self.survey.times[tind]
            Jtv = self.actMap.P*self.J.T.dot(v[:, tind])
            Jtvec += (
                self.PetaEtaDeriv(t, Jtv, adjoint=True) +
                self.PetaTauiDeriv(t, Jtv, adjoint=True) +
                self.PetaCDeriv(t, Jtv, adjoint=True)
                )

        return self.sign * Jtvec

    def getJtJdiag(self):
        """
        Compute JtJ using adjoint problem. Still we never form
        JtJ
        """
        ntime = len(self.survey.times)
        JtJdiag = np.zeros_like(self.model)
        for tind in range(ntime):
            t = self.survey.times[tind]
            Jtv = self.actMap.P*Utils.sdiag(1./self.mesh.vol)*self.J.T
            JtJdiag += (
                (self.PetaEtaDeriv(t, Jtv, adjoint=True)**2).sum(axis=1) +
                (self.PetaTauiDeriv(t, Jtv, adjoint=True)**2).sum(axis=1) +
                (self.PetaCDeriv(t, Jtv, adjoint=True)**2).sum(axis=1)
                )
        return JtJdiag

    def MfRhoIDeriv(self, u):
        """
            Derivative of :code:`MfRhoI` with respect to the model.
        """

        dMfRhoI_dI = -self.MfRhoI**2
        dMf_drho = self.mesh.getFaceInnerProductDeriv(self.rho)(u)
        drho_dlogrho = Utils.sdiag(self.rho)*self.actMap.P
        return dMfRhoI_dI * (dMf_drho * drho_dlogrho)

    # TODO: This should take a vector
    def MeSigmaDeriv(self, u):
        """
            Derivative of MeSigma with respect to the model
        """
        dsigma_dlogsigma = Utils.sdiag(self.sigma)*self.actMap.P
        return (
            self.mesh.getEdgeInnerProductDeriv(self.sigma)(u)
            * dsigma_dlogsigma
            )

    def MnSigmaDeriv(self, u):
        """
            Derivative of MnSigma with respect to the model
        """
        sigma = self.sigma
        vol = self.mesh.vol
        dsigma_dlogsigma = Utils.sdiag(sigma)*self.actMap.P
        return (
            Utils.sdiag(u)*self.mesh.aveN2CC.T *
            (Utils.sdiag(vol) * dsigma_dlogsigma)
                )

    def MccRhoIDeriv(self, u):
        """
            Derivative of MccRhoI with respect to the model
        """
        rho = self.rho
        vol = self.mesh.vol
        drho_dlogrho = Utils.sdiag(rho)*self.actMap.P
        return (
            Utils.sdiag(u.flatten()*vol*(-1./rho**2))*drho_dlogrho
            )


class Problem2D_CC(BaseSIPProblem_2D):
    """
    2.5D cell centered Spectral IP problem
    """

    _solutionType = 'phiSolution'
    _formulation = 'HJ'  # CC potentials means J is on faces
    fieldsPair = Fields_ky_CC
    sign = 1.

    def __init__(self, mesh, **kwargs):
        BaseSIPProblem_2D.__init__(self, mesh, **kwargs)
        self.setBC()
        if self.actinds is None:
            print ("You did not put Active indices")
            print ("So, set actMap = IdentityMap(mesh)")
            self.actinds = np.ones(mesh.nC, dtype=bool)

        self.actMap = Maps.InjectActiveCells(mesh, self.actinds, 0.)

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
        MfRhoIDeriv = self.MfRhoIDeriv
        MccRhoIDeriv = self.MccRhoIDeriv
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


class Problem2D_N(BaseSIPProblem_2D):
    """
    2.5D nodal Spectral IP problem
    """

    _solutionType = 'phiSolution'
    _formulation = 'EB'  # CC potentials means J is on faces
    fieldsPair = Fields_ky_N
    sign = -1.

    def __init__(self, mesh, **kwargs):
        BaseSIPProblem_2D.__init__(self, mesh, **kwargs)
        # self.setBC()
        if self.actinds is None:
            print ("You did not put Active indices")
            print ("So, set actMap = IdentityMap(mesh)")
            self.actinds = np.ones(mesh.nC, dtype=bool)

        self.actMap = Maps.InjectActiveCells(mesh, self.actinds, 0.)

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

    def getADeriv(self, ky, u, v, adjoint=False):

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
