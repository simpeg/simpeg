from SimPEG import Problem, Utils, Maps, Mesh
from SimPEG.EM.Base import BaseEMProblem
from SimPEG.EM.Static.DC.FieldsDC import FieldsDC, Fields_CC
from SimPEG.EM.Static.DC import Survey, BaseDCProblem, Problem3D_CC
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

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        return toDelete

    def evalq(self, Qv, vel):
        MfQviI = self.mesh.getFaceInnerProduct(1./Qv, invMat=True)
        Mf = self.mesh.getFaceInnerProduct()
        return self.Div*(Mf*(MfQviI*vel))


class Problem_CC(BaseSPProblem, Problem3D_CC):

    _solutionType = 'phiSolution'
    _formulation = 'HJ'  # CC potentials means J is on faces
    fieldsPair = Fields_CC
    modelType = None
    bc_type = "Mixed"
    coordinate_system = properties.StringChoice(
        "Type of coordinate system we are regularizing in",
        choices=['cartesian', 'spherical'],
        default='cartesian'
    )

    def __init__(self, mesh, **kwargs):
        BaseSPProblem.__init__(self, mesh, **kwargs)
        self.setBC()

    def getADeriv(self, u, v, adjoint= False):
        # We assume conductivity is known
        return Zero()

    def getRHSDeriv(self, src, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model
        """
        return src.evalDeriv(self, v=v, adjoint=adjoint)


class Problem_CC_Jstore(Problem_CC):
    """docstring for Problem_CC_jstore"""

    _S = None

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
            self._G = (self.Ainv * P.T).T * src.evalDeriv(
                self, v=Utils.sdiag(np.ones_like(self.model))
            )
            self.Ainv.clean()
            del self.Ainv
        return self._G

    def getJ(self, m, f=None):

        if self.coordinate_system == 'cartesian':
            return self.G
        else:
            self.model = m
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
            if self.verbose:
                print ("Compute S")
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

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = super(BaseDCProblem, self).deleteTheseOnModelUpdate
        if self._S is not None:
            toDelete += ['_S']
        return toDelete


class SurveySP_store(Survey):
    @Utils.count
    @Utils.requires('prob')
    def dpred(self, m=None, f=None):
        return self.prob.fields(m)


