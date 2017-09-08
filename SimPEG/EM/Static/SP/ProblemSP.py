from SimPEG import Problem, Utils, Maps, Mesh
from SimPEG.EM.Base import BaseEMProblem
from SimPEG.EM.Static.DC.FieldsDC import FieldsDC, Fields_CC
from SimPEG.EM.Static.DC import Survey, BaseDCProblem
from SimPEG.Utils import sdiag
import numpy as np
from SimPEG.Utils import Zero
from SimPEG.EM.Static.DC import getxBCyBC_CC
from SimPEG import Props


class BaseSPProblem(BaseDCProblem):

    h, hMap, hDeriv = Props.Invertible(
        "Hydraulic Head (m)"
    )

    q, qMap, qDeriv = Props.Invertible(
        "Streaming current source (A/m^3)"
    )

    jsx, jsxMap, jsxDeriv = Props.Invertible(
        "Streaming current density in x-direction (A/m^2)"
    )

    jsy, jsyMap, jsyDeriv = Props.Invertible(
        "Streaming current density in y-direction (A/m^2)"
    )

    jsz, jszMap, jszDeriv = Props.Invertible(
        "Streaming current density in z-direction (A/m^2)"
    )

    sigma = Props.PhysicalProperty(
        "Electrical conductivity (S/m)"
    )

    rho = Props.PhysicalProperty(
        "Electrical resistivity (Ohm m)"
    )

    Props.Reciprocal(sigma, rho)

    modelType = None
    surveyPair = Survey
    fieldsPair = FieldsDC
    # PropMap = SPPropMap
    Ainv = None
    sigma = None
    rho = None
    f = None
    Ainv = None

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        return toDelete

    def evalq(self, Qv, vel):
        MfQviI = self.mesh.getFaceInnerProduct(1./Qv, invMat=True)
        Mf = self.mesh.getFaceInnerProduct()
        return self.Div*(Mf*(MfQviI*vel))


class Problem_CC(BaseSPProblem):

    _solutionType = 'phiSolution'
    _formulation = 'HJ'  # CC potentials means J is on faces
    fieldsPair = Fields_CC
    modelType = None

    def __init__(self, mesh, **kwargs):
        BaseSPProblem.__init__(self, mesh, **kwargs)
        # if self.rho is None:
        #     raise Exception("Resistivity:rho needs to set when \
        #                      initializing SPproblem")
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
        return A

    def getADeriv(self, u, v, adjoint= False):
        # We assume conductivity is known
        return Zero()

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
        return src.evalDeriv(self, v=v, adjoint=adjoint)

    def setBC(self):
        if self.mesh.dim==3:
            fxm,fxp,fym,fyp,fzm,fzp = self.mesh.faceBoundaryInd
            gBFxm = self.mesh.gridFx[fxm, :]
            gBFxp = self.mesh.gridFx[fxp, :]
            gBFym = self.mesh.gridFy[fym, :]
            gBFyp = self.mesh.gridFy[fyp, :]
            gBFzm = self.mesh.gridFz[fzm, :]
            gBFzp = self.mesh.gridFz[fzp, :]

            # Setup Mixed B.C (alpha, beta, gamma)
            temp_xm, temp_xp = np.ones_like(gBFxm[:,0]), np.ones_like(gBFxp[:,0])
            temp_ym, temp_yp = np.ones_like(gBFym[:,1]), np.ones_like(gBFyp[:,1])
            temp_zm, temp_zp = np.ones_like(gBFzm[:,2]), np.ones_like(gBFzp[:,2])

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
            beta =  [beta_xm, beta_xp, beta_ym, beta_yp, beta_zm, beta_zp]
            gamma = [gamma_xm, gamma_xp, gamma_ym, gamma_yp, gamma_zm, gamma_zp]

        elif self.mesh.dim==2:

            fxm,fxp,fym,fyp = self.mesh.faceBoundaryInd
            gBFxm = self.mesh.gridFx[fxm,:]
            gBFxp = self.mesh.gridFx[fxp,:]
            gBFym = self.mesh.gridFy[fym,:]
            gBFyp = self.mesh.gridFy[fyp,:]

            # Setup Mixed B.C (alpha, beta, gamma)
            temp_xm, temp_xp = np.ones_like(gBFxm[:,0]), np.ones_like(gBFxp[:,0])
            temp_ym, temp_yp = np.ones_like(gBFym[:,1]), np.ones_like(gBFyp[:,1])

            alpha_xm, alpha_xp = temp_xm*0., temp_xp*0.
            alpha_ym, alpha_yp = temp_ym*0., temp_yp*0.

            beta_xm, beta_xp = temp_xm, temp_xp
            beta_ym, beta_yp = temp_ym, temp_yp

            gamma_xm, gamma_xp = temp_xm*0., temp_xp*0.
            gamma_ym, gamma_yp = temp_ym*0., temp_yp*0.

            alpha = [alpha_xm, alpha_xp, alpha_ym, alpha_yp]
            beta =  [beta_xm, beta_xp, beta_ym, beta_yp]
            gamma = [gamma_xm, gamma_xp, gamma_ym, gamma_yp]

        x_BC, y_BC = getxBCyBC_CC(self.mesh, alpha, beta, gamma)
        V = self.Vol
        self.Div = V * self.mesh.faceDiv
        P_BC, B = self.mesh.getBCProjWF_simple()
        M = B*self.mesh.aveCC2F
        self.Grad = self.Div.T - P_BC*Utils.sdiag(y_BC)*M


class Problem_CC_Jstore(object):

    """docstring for Problem_CC_jstore"""

    def Jvec(self, m, v, f=None):
        return self.G * v

    def Jtvec(self, m, v, f=None):
        return self.G.T * v




