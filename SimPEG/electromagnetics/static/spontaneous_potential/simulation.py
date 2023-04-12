# from SimPEG import Utils
# from SimPEG.EM.Static.DC.FieldsDC import FieldsDC, Fields3DCellCentered
# from SimPEG.EM.Static.DC import Survey, BaseDCProblem, Simulation3DCellCentered
import numpy as np
import scipy.sparse as sp

# from SimPEG.Utils import Zero
# from SimPEG import Props

from .... import props, maps
from ....base import BasePDESimulation, with_property_mass_matrices
from ..resistivity import Simulation3DCellCentered as DC_3D_CC
from ....survey import BaseSurvey
from .sources import StreamingCurrents


class Simulation3DCellCentered(DC_3D_CC):
    q, qMap, qDeriv = Props.Invertible("Charge density accumulation rate (C/(s m^3))")

    def __init__(
        self, mesh, survey=None, sigma=None, rho=None, q=None, qMap=None, **kwargs
    ):
        if sigma is None:
            if rho is None:
                raise ValueError("Must set either conductivity or resistivity.")
        else:
            if rho is not None:
                raise ValueError("Cannot set both conductivity and resistivity.")
        super().__init__(
            mesh=mesh,
            survey=survey,
            sigma=sigma,
            rho=rho,
            sigmaMap=None,
            rhoMap=None,
            **kwargs
        )
        self.q = q
        self.qMap = qMap

    def getRHS(self):
        return self.Vol @ self.q

    def getRHSDeriv(self, source, v, adjoint=False):
        if adjoint:
            return self.qDeriv.T @ (self.Vol @ v)
        return self.Vol @ (self.qDeriv @ v)

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        return toDelete

    # def evalq(self, Qv, vel):
    #     MfQviI = self.mesh.get_face_inner_product(1.0 / Qv, invert_matrix=True)
    #     Mf = self.mesh.get_face_inner_product()
    #     return self.Div * (Mf * (MfQviI * vel))


class CurrentDensityMap(maps.LinearMap):
    r"""Maps current density to charge density accumulation rate.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh

    Notes
    -----
    .. math::
        q = \nabla \cdot \vec{j}
    """

    def __init__(self, mesh):
        A = -mesh.face_divergence @ mesh.AvgCCV2F
        super().__init__(A)


class HydraulicHeadMap(maps.LinearMap):
    """Maps hydraulic head to charge density accumulation rate.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
    L : array_like
        Cross coupling property model (units????)

    Notes
    -----
    .. math::
        q = \nabla \cdot L \nabla h
    """

    def __init__(self, mesh, L):
        div = mesh.face_divergence
        MfLiI = mesh.get_face_inner_product(L, invert_model=True, invert_matrix=True)
        A = div.T @ MfLiI @ div @ sdiag(self.mesh.cell_volumes)
        super().__init__(A)


# class Problem_CC(BaseSPProblem, Simulation3DCellCentered):
#     _solutionType = "phiSolution"
#     _formulation = "HJ"  # CC potentials means J is on faces
#     fieldsPair = Fields3DCellCentered
#     modelType = None
#     bc_type = "Mixed"
#     # coordinate_system = StringChoice(
#     #     "Type of coordinate system we are regularizing in",
#     #     choices=["cartesian", "spherical"],
#     #     default="cartesian",
#     # )
#
#     def __init__(self, mesh, **kwargs):
#         BaseSPProblem.__init__(self, mesh, **kwargs)
#         self.setBC()
#
#     def getADeriv(self, u, v, adjoint=False):
#         # We assume conductivity is known
#         return Zero()
#
#     def getRHSDeriv(self, src, v, adjoint=False):
#         """
#         Derivative of the right hand side with respect to the model
#         """
#         return src.evalDeriv(self, v=v, adjoint=adjoint)
#
#
# class Problem_CC_Jstore(Problem_CC):
#     """docstring for Problem_CC_jstore"""
#
#     _S = None
#
#     @property
#     def G(self):
#         """
#         Inverse of :code:`_G`
#         """
#         if getattr(self, "_G", None) is None:
#             A = self.getA()
#             self.Ainv = self.solver(A, **self.solver_opts)
#             src = self.survey.source_list[0]
#             rx = src.receiver_list[0]
#             P = rx.getP(self.mesh, "CC").toarray()
#             src = self.survey.source_list[0]
#             self._G = (self.Ainv * P.T).T * src.evalDeriv(
#                 self, v=Utils.sdiag(np.ones_like(self.model))
#             )
#             self.Ainv.clean()
#             del self.Ainv
#         return self._G
#
#     def getJ(self, m, f=None):
#         if self.coordinate_system == "cartesian":
#             return self.G
#         else:
#             self.model = m
#             return self.G * self.S
#
#     def Jvec(self, m, v, f=None):
#         self.model = m
#
#         if self.coordinate_system == "cartesian":
#             return self.G.dot(v)
#         else:
#             return np.dot(self.G, self.S.dot(v))
#
#     def Jtvec(self, m, v, f=None):
#         self.model = m
#
#         if self.coordinate_system == "cartesian":
#             return self.G.T.dot(v)
#         else:
#             return self.S.T * (self.G.T.dot(v))
#
#     @Utils.count
#     def fields(self, m):
#         self.model = m
#
#         if self.coordinate_system == "spherical":
#             m = Utils.mat_utils.atp2xyz(m)
#
#         return self.G.dot(m)
#
#     @property
#     def S(self):
#         """
#         Derivatives for the spherical transformation
#         """
#         if getattr(self, "_S", None) is None:
#             if self.verbose:
#                 print("Compute S")
#             if self.model is None:
#                 raise Exception("Requires a model")
#
#             # Assume it is vector model in spherical coordinates
#             nC = int(self.model.shape[0] / 3)
#
#             a = self.model[:nC]
#             t = self.model[nC : 2 * nC]
#             p = self.model[2 * nC :]
#
#             Sx = sp.hstack(
#                 [
#                     sp.diags(np.cos(t) * np.cos(p), 0),
#                     sp.diags(-a * np.sin(t) * np.cos(p), 0),
#                     sp.diags(-a * np.cos(t) * np.sin(p), 0),
#                 ]
#             )
#
#             Sy = sp.hstack(
#                 [
#                     sp.diags(np.cos(t) * np.sin(p), 0),
#                     sp.diags(-a * np.sin(t) * np.sin(p), 0),
#                     sp.diags(a * np.cos(t) * np.cos(p), 0),
#                 ]
#             )
#
#             Sz = sp.hstack(
#                 [
#                     sp.diags(np.sin(t), 0),
#                     sp.diags(a * np.cos(t), 0),
#                     sp.csr_matrix((nC, nC)),
#                 ]
#             )
#
#             self._S = sp.vstack([Sx, Sy, Sz])
#
#         return self._S
#
#     @property
#     def deleteTheseOnModelUpdate(self):
#         toDelete = super().deleteTheseOnModelUpdate
#         if self._S is not None:
#             toDelete = toDelete + ["_S"]
#         return toDelete
#
#


class Survey(BaseSurvey):
    @property
    def source_list(self):
        return self._source_list

    @source_list.setter
    def source_list(self, new_list):
        self._source_list = validate_list_of_types(
            "source_list",
            new_list,
            StreamingCurrents,
        )
