from discretize import TensorMesh
from SimPEG import utils, maps, props
from SimPEG.utils import Zero, sdiag
from .survey import Survey
from ...base import BaseEMSimulation
from ..resistivity import BaseDCSimulation, FieldsDC, Fields3DCellCentered, getxBCyBC_CC
from ..resistivity import Simulation3DCellCentered as DCSimulation3DCellCentered

import numpy as np
import scipy.sparse as sp
import properties

class BaseSimulation(BaseDCSimulation):

    _G = None
    _S = None
    fieldsPair = FieldsDC

    survey = properties.Instance("an SP survey object", Survey, required=True)

    sigma = props.PhysicalProperty("Electrical conductivity (S/m)")

    rho = props.PhysicalProperty("Electrical resistivity (Ohm m)")

    props.Reciprocal(sigma, rho)

    @property
    def V(self):
        """
        Sparse cell volume matrix
        """
        if getattr(self, "_V", None) is None:
            self._V = utils.sdiag(self.mesh.vol)
        return self._V

    @property
    def Pac(self):
        """
        diagonal matrix that nulls out inactive cells

        :rtype: scipy.sparse.csr_matrix
        :return: active cell diagonal matrix
        """
        if getattr(self, "_Pac", None) is None:
            if self.indActive is None:
                self._Pac = utils.speye(self.mesh.nC)
            else:
                e = np.zeros(self.mesh.nC)
                e[self.indActive] = 1.0
                # self._Pac = utils.speye(self.mesh.nC)[:, self.indActive]
                self._Pac = utils.sdiag(e)
        return self._Pac


    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        return toDelete

    def evalq(self, Qv, vel):
        MfQviI = self.mesh.getFaceInnerProduct(1.0 / Qv, invMat=True)
        Mf = self.mesh.getFaceInnerProduct()
        return self.Div * (Mf * (MfQviI * vel))
    
    def dpred(self, m=None, f=None):
        return self.fields(m)

    @property
    def Pafx(self):
        """
        diagonal matrix that nulls out inactive x-faces
        to full modelling space (ie. nFx x nindActive_Fx )

        :rtype: scipy.sparse.csr_matrix
        :return: active face-x diagonal matrix
        """
        if getattr(self, "_Pafx", None) is None:
            if self.indActive is None:
                self._Pafx = utils.speye(self.mesh.nFx)
            else:
                indActive_Fx = self.mesh.aveFx2CC.T * self.indActive >= 1
                e = np.zeros(self.mesh.nFx)
                e[indActive_Fx] = 1.0
                self._Pafx = utils.sdiag(e)
        return self._Pafx

    @property
    def Pafy(self):
        """
        diagonal matrix that nulls out inactive y-faces
        to full modelling space (ie. nFy x nindActive_Fy )

        :rtype: scipy.sparse.csr_matrix
        :return: active face-y diagonal matrix
        """
        if getattr(self, "_Pafy", None) is None:
            if self.indActive is None:
                self._Pafy = utils.speye(self.mesh.nFy)
            else:
                indActive_Fy = (self.mesh.aveFy2CC.T * self.indActive) >= 1
                e = np.zeros(self.mesh.nFy)
                e[indActive_Fy] = 1.0
                self._Pafy = utils.sdiag(e)
        return self._Pafy

    @property
    def Pafz(self):
        """
        diagonal matrix that nulls out inactive z-faces
        to full modelling space (ie. nFz x nindActive_Fz )

        :rtype: scipy.sparse.csr_matrix
        :return: active face-z diagonal matrix
        """
        if getattr(self, "_Pafz", None) is None:
            if self.indActive is None:
                self._Pafz = utils.speye(self.mesh.nFz)
            else:
                indActive_Fz = (self.mesh.aveFz2CC.T * self.indActive) >= 1
                e = np.zeros(self.mesh.nFz)
                e[indActive_Fz] = 1.0
                self._Pafz = utils.sdiag(e)
        return self._Pafz


    def getADeriv(self, u, v, adjoint=False):
        # We assume conductivity is known
        return Zero()

    def getRHSDeriv(self, src, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model
        """
        return self.get_source_term_derivative(self, v=v, adjoint=adjoint)

    @property
    def G(self):
        """
            Inverse of :code:`_G`
        """
        if getattr(self, "_G", None) is None:
            A = self.getA()
            self.Ainv = self.Solver(A, **self.solverOpts)
            src = self.survey.source_list[0]
            rx = src.receiver_list[0]
            P = rx.getP(self.mesh, "CC").toarray()
            self._G = (self.Ainv * P.T).T * self.get_source_term_derivative(
                v=utils.sdiag(np.ones_like(self.model))
            )
            self.Ainv.clean()
            del self.Ainv
        return self._G

    def getJ(self, m, f=None):

        if self.coordinate_system == "cartesian":
            return self.G
        else:
            self.model = m
            return self.G * self.S

    def Jvec(self, m, v, f=None):

        self.model = m

        if self.coordinate_system == "cartesian":
            return self.G.dot(v)
        else:
            return np.dot(self.G, self.S.dot(v))

    def Jtvec(self, m, v, f=None):

        self.model = m

        if self.coordinate_system == "cartesian":
            return self.G.T.dot(v)
        else:
            return self.S.T * (self.G.T.dot(v))

    @utils.count
    def fields(self, m):

        self.model = m

        if self.coordinate_system == "spherical":
            m = utils.mat_utils.atp2xyz(m)

        return self.G.dot(m)

    @property
    def S(self):
        """
            Derivatives for the spherical transformation
        """
        if getattr(self, "_S", None) is None:
            if self.verbose:
                print("Compute S")
            if self.model is None:
                raise Exception("Requires a model")

            # Assume it is vector model in spherical coordinates
            nC = int(self.model.shape[0] / 3)

            a = self.model[:nC]
            t = self.model[nC : 2 * nC]
            p = self.model[2 * nC :]

            Sx = sp.hstack(
                [
                    sp.diags(np.cos(t) * np.cos(p), 0),
                    sp.diags(-a * np.sin(t) * np.cos(p), 0),
                    sp.diags(-a * np.cos(t) * np.sin(p), 0),
                ]
            )

            Sy = sp.hstack(
                [
                    sp.diags(np.cos(t) * np.sin(p), 0),
                    sp.diags(-a * np.sin(t) * np.sin(p), 0),
                    sp.diags(a * np.cos(t) * np.cos(p), 0),
                ]
            )

            Sz = sp.hstack(
                [
                    sp.diags(np.sin(t), 0),
                    sp.diags(a * np.cos(t), 0),
                    sp.csr_matrix((nC, nC)),
                ]
            )

            self._S = sp.vstack([Sx, Sy, Sz])

        return self._S

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = super(BaseDCSimulation, self).deleteTheseOnModelUpdate
        if self._S is not None:
            toDelete += ["_S"]
        return toDelete






class SimulationCurrentDensity(BaseSimulation, DCSimulation3DCellCentered):

    jsx, jsxMap, jsxDeriv = props.Invertible(
        "Streaming current density in x-direction (A/m^2)"
    )

    jsy, jsyMap, jsyDeriv = props.Invertible(
        "Streaming current density in y-direction (A/m^2)"
    )

    jsz, jszMap, jszDeriv = props.Invertible(
        "Streaming current density in z-direction (A/m^2)"
    )

    _solutionType = "phiSolution"
    _formulation = "HJ"  # CC potentials means J is on faces
    fieldsPair = Fields3DCellCentered
    bc_type = "Mixed"
    coordinate_system = properties.StringChoice(
        "Type of coordinate system we are regularizing in",
        choices=["cartesian", "spherical"],
        default="cartesian",
    )

    def __init__(self, mesh, **kwargs):
        BaseSimulation.__init__(self, mesh, **kwargs)
        self.setBC()

        # This is for setting a Neuman condition on the topographic faces
        if self.indActive is None:
            self.indActive = np.ones(self.mesh.nC, dtype=bool)
        
        self.Grad = -sp.vstack(
            (
                self.Pafx * self.mesh.faceDivx.T * self.V * self.Pac,
                self.Pafy * self.mesh.faceDivy.T * self.V * self.Pac,
                self.Pafz * self.mesh.faceDivz.T * self.V * self.Pac,
                )
            )


    def getq_from_j(self, j):
        q = self.Grad.T * self.mesh.aveCCV2F * j
        return q

    def get_source_term(self):
        """

            Computing source term using:

            - Hydraulic head: h
            - Cross coupling coefficient: L

            .. math::

                -\nabla \cdot \vec{j}^s = \nabla \cdot L \nabla \phi \\

        """

        q = (
            self.Grad.T
            * self.mesh.aveCCV2F
            * np.r_[self.jsx, self.jsy, self.jsz]
        )

        return q


    

    def get_source_term_derivative(self, v=None, adjoint=False):
        """Computing the derivative of the source terms with respect to the model"""

        if adjoint:
            jsDeriv = sp.vstack((self.jsxDeriv, self.jsyDeriv, self.jszDeriv))
            srcDeriv = jsDeriv.T * self.mesh.aveCCV2F.T * (self.Grad * v)
        else:
            jsDeriv = sp.vstack((self.jsxDeriv, self.jsyDeriv, self.jszDeriv))
            srcDeriv = self.Grad.T * self.mesh.aveCCV2F * (jsDeriv * v)

        return srcDeriv

    








class SimulationCurrentSource(BaseSimulation, DCSimulation3DCellCentered):

    q, qMap, qDeriv = props.Invertible("Streaming current source (A/m^3)")

    _solutionType = "phiSolution"
    _formulation = "HJ"  # CC potentials means J is on faces
    fieldsPair = Fields3DCellCentered
    bc_type = "Mixed"
    coordinate_system = properties.StringChoice(
        "Type of coordinate system we are regularizing in",
        choices=["cartesian", "spherical"],
        default="cartesian",
    )

    def __init__(self, mesh, **kwargs):
        BaseSimulation.__init__(self, mesh, **kwargs)
        self.setBC()

    def get_source_term(self):
        """

            Computing source term using:

            - Hydraulic head: h
            - Cross coupling coefficient: L

            .. math::

                -\nabla \cdot \vec{j}^s = \nabla \cdot L \nabla \phi \\

        """
        
        return self.V * self.q

    def get_source_term_derivative(self, v=None, adjoint=False):
        """Computing the derivative of the source terms with respect to the model"""

        if adjoint:
            srcDeriv = self.qDeriv.T * (self.V * v)
        else:
            srcDeriv = self.V * (self.qDeriv * v)
                
        return srcDeriv




class SimulationHydrolicHead(BaseSimulation, DCSimulation3DCellCentered):

    h, hMap, hDeriv = props.Invertible("Hydraulic Head (m)")

    L = props.PhysicalProperty("Cross-correlation coefficients")


    _solutionType = "phiSolution"
    _formulation = "HJ"  # CC potentials means J is on faces
    fieldsPair = Fields3DCellCentered
    bc_type = "Mixed"
    coordinate_system = properties.StringChoice(
        "Type of coordinate system we are regularizing in",
        choices=["cartesian", "spherical"],
        default="cartesian",
    )

    def __init__(self, mesh, **kwargs):
        BaseSimulation.__init__(self, mesh, **kwargs)
        self.setBC()


    @property
    def MfLi(self):
        """
        Mass matrix for cross-correlation coefficients`
        """
        if getattr(self, "_MfLi", None) is None:
            self._MfLi = self.mesh.getFaceInnerProduct(1.0 / self.L)
        return self._MfLi

    @property
    def MfLiI(self):
        """
        Inverse of mass matrix for cross-correlation coefficients
        """
        if getattr(self, "_MfLiI", None) is None:
            self._MfLiI = self.mesh.getFaceInnerProduct(1.0 / self.L, invMat=True)
        return self._MfLiI

    def get_source_term(self):
        """

            Computing source term using:

            - Hydraulic head: h
            - Cross coupling coefficient: L

            .. math::

                -\nabla \cdot \vec{j}^s = \nabla \cdot L \nabla \phi \\

        """

        return self.Grad.T * self.MfLiI * self.Grad * self.h
            

    def get_source_term_derivative(self, v=None, adjoint=False):
        """Computing the derivative of the source terms with respect to the model"""

        if adjoint:
            srcDeriv = (
                self.hDeriv.T * self.Grad.T * self.MfLiI.T * (self.Grad * v)
            )
        else:
            srcDeriv = self.Grad.T * self.MfLiI * self.Grad * (self.hDeriv * v)
                
        return srcDeriv








# class SurveySP_store(Survey):
#     @utils.count
#     @utils.requires("prob")
#     def dpred(self, m=None, f=None):
#         return self.fields(m)
