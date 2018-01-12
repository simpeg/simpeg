from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from SimPEG import Utils
from SimPEG.EM.Base import BaseEMProblem
from .SurveyDC import Survey_ky
from .FieldsDC_2D import Fields_ky, Fields_ky_CC, Fields_ky_N
import numpy as np
from SimPEG.Utils import Zero
from .BoundaryUtils import getxBCyBC_CC


class BaseDCProblem_2D(BaseEMProblem):
    """
    Base 2.5D DC problem
    """

    surveyPair = Survey_ky
    fieldsPair = Fields_ky  # SimPEG.EM.Static.Fields_2D
    nky = 15
    kys = np.logspace(-4, 1, nky)
    Ainv = [None for i in range(nky)]
    nT = nky  # Only for using TimeFields

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

    def Jvec(self, m, v, f=None):

        if f is None:
            f = self.fields(m)

        self.model = m

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
        if f is None:
            f = self.fields(m)

        self.model = m

        # Ensure v is a data object.
        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        Jtv = np.zeros(m.size, dtype=float)

        # Assume y=0.
        # This needs some thoughts to implement in general when src is dipole
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
                    df_duTFun = getattr(f, '_{0!s}Deriv'.format(rx.projField),
                                        None)
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


class Problem2D_CC(BaseDCProblem_2D):
    """
    2.5D cell centered DC problem
    """

    _solutionType = 'phiSolution'
    _formulation = 'HJ'  # CC potentials means J is on faces
    fieldsPair = Fields_ky_CC

    def __init__(self, mesh, **kwargs):
        BaseDCProblem_2D.__init__(self, mesh, **kwargs)
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


class Problem2D_N(BaseDCProblem_2D):
    """
    2.5D nodal DC problem
    """

    _solutionType = 'phiSolution'
    _formulation = 'EB'  # CC potentials means J is on faces
    fieldsPair = Fields_ky_N

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

        # Handling Null space of A
        A[0, 0] = A[0, 0] + 1.
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
