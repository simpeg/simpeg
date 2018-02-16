from SimPEG import Problem, Utils, Maps, Mesh
from SimPEG.EM.Base import BaseEMProblem
from SimPEG.EM.Static.DC.FieldsDC import FieldsDC, Fields_CC
from SimPEG.EM.Static.DC import Survey, BaseDCProblem
from SimPEG.Utils import sdiag
import numpy as np
import scipy.sparse as sp
from SimPEG.Utils import Zero
from SimPEG.EM.Static.DC import getxBCyBC_CC
from SimPEG import Props
import properties


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
    coordinate_system = properties.StringChoice(
    "Type of coordinate system we are regularizing in",
    choices=['cartesian', 'spherical'],
    default='cartesian' )

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


class Problem_CC_Jstore(Problem_CC):
    """docstring for Problem_CC_jstore"""
    @property
    def G(self):
        """
            Inverse of :code:`_G`
        """
        if getattr(self, '_G', None) is None:
            A = self.getA()
            self.Ainv = self.Solver(A, **self.solverOpts)
            src = self.survey.srcList[0]
            rx = src.rxList[0]
            P = rx.getP(self.mesh, "CC").toarray()
            src = self.survey.srcList[0]
            self._G = ((self.Ainv * P.T).T) * src.evalDeriv(self, v=Utils.sdiag(np.ones_like(self.model)))
            self.Ainv.clean()
        return self._G

    def getJ(self, m, f=None):

        if self.coordinate_system == 'cartesian':
            return self.G
        else:
            return self.G * self.S

    def Jvec(self, m, v, f=None):

        self.model = m

        if self.coordinate_system == 'cartesian':
            return self.G.dot(v)
        else:
            return np.dot(self.G, self.S.dot(v))

    def Jtvec(self, m, v, f=None):

        self.model = m

        if self.coordinate_system == 'cartesian':
            return self.G.T.dot(v)
        else:
            return self.S.T*(self.G.T.dot(v))

    @Utils.count
    def fields(self, m):

        self.model = m

        if self.coordinate_system == 'spherical':
            m = Utils.matutils.atp2xyz(m)

        return self.G.dot(m)

    @property
    def S(self):
        """
            Derivatives for the spherical transformation
        """
        if getattr(self, '_S', None) is None:

            if self.model is None:
                raise Exception('Requires a model')

            # Assume it is vector model in spherical coordinates
            nC = int(self.model.shape[0]/3)

            a = self.model[:nC]
            t = self.model[nC:2*nC]
            p = self.model[2*nC:]

            Sx = sp.hstack([sp.diags(np.cos(t)*np.cos(p), 0),
                            sp.diags(-a*np.sin(t)*np.cos(p), 0),
                            sp.diags(-a*np.cos(t)*np.sin(p), 0)])

            Sy = sp.hstack([sp.diags(np.cos(t)*np.sin(p), 0),
                            sp.diags(-a*np.sin(t)*np.sin(p), 0),
                            sp.diags(a*np.cos(t)*np.cos(p), 0)])

            Sz = sp.hstack([sp.diags(np.sin(t), 0),
                            sp.diags(a*np.cos(t), 0),
                            sp.csr_matrix((nC, nC))])

            self._S = sp.vstack([Sx, Sy, Sz])

        return self._S


class SurveySP_store(Survey):
    @Utils.count
    @Utils.requires('prob')
    def dpred(self, m=None, f=None):
        return self.prob.fields(m)


