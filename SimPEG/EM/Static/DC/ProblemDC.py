from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from SimPEG import Utils
from SimPEG.EM.Base import BaseEMProblem
from .SurveyDC import Survey
from .FieldsDC import FieldsDC, Fields_CC, Fields_N
import numpy as np
import scipy as sp
from SimPEG.Utils import Zero
from .BoundaryUtils import getxBCyBC_CC


class BaseDCProblem(BaseEMProblem):
    """
    Base DC Problem
    """
    surveyPair = Survey
    fieldsPair = FieldsDC
    Ainv = None

    def fields(self, m=None):
        if m is not None:
            self.model = m

        if self.Ainv is not None:
            self.Ainv.clean()

        f = self.fieldsPair(self.mesh, self.survey)
        A = self.getA()
        self.Ainv = self.Solver(A, **self.solverOpts)
        RHS = self.getRHS()
        u = self.Ainv * RHS
        Srcs = self.survey.srcList
        f[Srcs, self._solutionType] = u
        return f

    def Jvec(self, m, v, f=None):

        if f is None:
            f = self.fields(m)

        self.model = m

        # Jv = self.dataPair(self.survey)  # same size as the data
        Jv = []

        A = self.getA()

        for src in self.survey.srcList:
            u_src = f[src, self._solutionType]  # solution vector
            dA_dm_v = self.getADeriv(u_src, v)
            dRHS_dm_v = self.getRHSDeriv(src, v)
            du_dm_v = self.Ainv * (- dA_dm_v + dRHS_dm_v)

            for rx in src.rxList:
                df_dmFun = getattr(f, '_{0!s}Deriv'.format(rx.projField), None)
                df_dm_v = df_dmFun(src, du_dm_v, v, adjoint=False)
                Jv.append(rx.evalDeriv(src, self.mesh, f, df_dm_v))
                # Jv[src, rx] = rx.evalDeriv(src, self.mesh, f, df_dm_v)
        # return Utils.mkvc(Jv)
        return np.hstack(Jv)

    def Jtvec(self, m, v, f=None):
        if f is None:
            f = self.fields(m)

        self.model = m

        # Ensure v is a data object.
        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        Jtv = np.zeros(m.size)
        AT = self.getA()

        for src in self.survey.srcList:
            u_src = f[src, self._solutionType]
            for rx in src.rxList:
                # wrt f, need possibility wrt m
                PTv = rx.evalDeriv(src, self.mesh, f, v[src, rx], adjoint=True)
                df_duTFun = getattr(f, '_{0!s}Deriv'.format(rx.projField),
                                    None)
                df_duT, df_dmT = df_duTFun(src, None, PTv, adjoint=True)

                ATinvdf_duT = self.Ainv * df_duT

                dA_dmT = self.getADeriv(u_src, ATinvdf_duT, adjoint=True)
                dRHS_dmT = self.getRHSDeriv(src, ATinvdf_duT, adjoint=True)
                du_dmT = -dA_dmT + dRHS_dmT
                Jtv += (df_dmT + du_dmT).astype(float)

        return Utils.mkvc(Jtv)

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


class Problem3D_CC(BaseDCProblem):
    """
    3D cell centered DC problem
    """

    _solutionType = 'phiSolution'
    _formulation = 'HJ'  # CC potentials means J is on faces
    fieldsPair = Fields_CC
    bc_type = 'Neumann'

    def __init__(self, mesh, **kwargs):

        BaseDCProblem.__init__(self, mesh, **kwargs)
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
                A[0,jj] = 0.

            A[0, 0] = 1./Vol[0]

        # I think we should deprecate this for DC problem.
        # if self._makeASymmetric is True:
        #     return V.T * A
        return A

    def getADeriv(self, u, v, adjoint=False):

        D = self.Div
        G = self.Grad
        MfRhoIDeriv = self.MfRhoIDeriv

        if adjoint:
            return(MfRhoIDeriv(G * u).T) * (D.T * v)

        return D * (MfRhoIDeriv(G * u) * v)

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

    def setBC(self):
        if self.mesh._meshType == "TREE":
            if(self.bc_type == 'Neumann'):
                raise NotImplementedError()
            elif(self.bc_type == 'Dirchlet'):
                print('Homogeneous Dirchlet is the natural BC for this CC discretization.')
                self.Div = Utils.sdiag(self.mesh.vol) * self.mesh.faceDiv
                self.Grad = self.Div.T

        else:

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

                elif(self.bc_type == 'Dirchlet'):
                    if self.verbose:
                        print('Setting BC to Dirchlet.')
                    alpha_xm, alpha_xp = temp_xm, temp_xp
                    alpha_ym, alpha_yp = temp_ym, temp_yp
                    alpha_zm, alpha_zp = temp_zm, temp_zp

                    beta_xm, beta_xp = temp_xm*0, temp_xp*0
                    beta_ym, beta_yp = temp_ym*0, temp_yp*0
                    beta_zm, beta_zp = temp_zm*0, temp_zp*0

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


class Problem3D_N(BaseDCProblem):
    """
    3D nodal DC problem
    """

    _solutionType = 'phiSolution'
    _formulation = 'EB'  # N potentials means B is on faces
    fieldsPair = Fields_N

    def __init__(self, mesh, **kwargs):
        BaseDCProblem.__init__(self, mesh, **kwargs)

    def getA(self):
        """

        Make the A matrix for the cell centered DC resistivity problem

        A = G.T MeSigma G

        """

        MeSigma = self.MeSigma
        Grad = self.mesh.nodalGrad
        A = Grad.T * MeSigma * Grad

        Vol = self.mesh.vol

        # Handling Null space of A
        I, J, V = sp.sparse.find(A[0,:])
        for jj in J:
            A[0,jj] = 0.

        A[0, 0] = 1./Vol[0]

        return A

    def getADeriv(self, u, v, adjoint=False):
        """

        Product of the derivative of our system matrix with respect to the
        model and a vector

        """
        Grad = self.mesh.nodalGrad
        if not adjoint:
            return Grad.T*(self.MeSigmaDeriv(Grad*u)*v)
        elif adjoint:
            return self.MeSigmaDeriv(Grad*u).T * (Grad*v)

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
