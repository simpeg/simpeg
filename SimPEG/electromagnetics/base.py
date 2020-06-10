from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import properties
from scipy.constants import mu_0
import numpy as np

from ..data import Data
from ..maps import IdentityMap
from ..simulation import BaseSimulation
from ..survey import BaseSurvey, BaseSrc
from ..utils import sdiag, Zero, mkvc
from .. import props

__all__ = ["BaseEMSimulation", "BaseEMSrc"]


###############################################################################
#                                                                             #
#                             Base EM Simulation                                 #
#                                                                             #
###############################################################################


class BaseEMSimulation(BaseSimulation):

    sigma, sigmaMap, sigmaDeriv = props.Invertible("Electrical conductivity (S/m)")

    rho, rhoMap, rhoDeriv = props.Invertible("Electrical resistivity (Ohm m)")

    props.Reciprocal(sigma, rho)

    mu = props.PhysicalProperty("Magnetic Permeability (H/m)", default=mu_0)
    mui = props.PhysicalProperty("Inverse Magnetic Permeability (m/H)")

    props.Reciprocal(mu, mui)

    # mapPair = IdentityMap  #: Type of mapping to pair with

    verbose = False
    storeInnerProduct = True

    ####################################################
    # Make A Symmetric
    ####################################################
    @property
    def _makeASymmetric(self):
        if getattr(self, "__makeASymmetric", None) is None:
            self.__makeASymmetric = True
        return self.__makeASymmetric

    ####################################################
    # Mass Matrices
    ####################################################
    @property
    def _clear_on_mu_update(self):
        """
        These matrices are deleted if there is an update to the permeability
        model
        """
        return ["_MeMu", "_MeMuI", "_MfMui", "_MfMuiI", "_MfMuiDeriv", "_MeMuDeriv"]

    @property
    def _clear_on_sigma_update(self):
        """
        These matrices are deleted if there is an update to the conductivity
        model
        """
        return [
            "_MeSigma",
            "_MeSigmaI",
            "_MfRho",
            "_MfRhoI",
            "_MeSigmaDeriv",
            "_MfRhoDeriv",
        ]

    @property
    def deleteTheseOnModelUpdate(self):
        """
        matrices to be deleted if the model maps for conductivity and/or
        permeability are updated
        """
        toDelete = []
        if self.sigmaMap is not None or self.rhoMap is not None:
            toDelete += self._clear_on_sigma_update

        if hasattr(self, "muMap") or hasattr(self, "muiMap"):
            if self.muMap is not None or self.muiMap is not None:
                toDelete += self._clear_on_mu_update
        return toDelete

    @properties.observer("mu")
    def _clear_mu_mats_on_mu_update(self, change):
        if change["previous"] is change["value"]:
            return
        if (
            isinstance(change["previous"], np.ndarray)
            and isinstance(change["value"], np.ndarray)
            and np.allclose(change["previous"], change["value"])
        ):
            return
        for mat in self._clear_on_mu_update:
            if hasattr(self, mat):
                delattr(self, mat)

    @properties.observer("mui")
    def _clear_mu_mats_on_mui_update(self, change):
        if change["previous"] is change["value"]:
            return
        if (
            isinstance(change["previous"], np.ndarray)
            and isinstance(change["value"], np.ndarray)
            and np.allclose(change["previous"], change["value"])
        ):
            return
        for mat in self._clear_on_mu_update:
            if hasattr(self, mat):
                delattr(self, mat)

    @properties.observer("sigma")
    def _clear_sigma_mats_on_sigma_update(self, change):
        if change["previous"] is change["value"]:
            return
        if (
            isinstance(change["previous"], np.ndarray)
            and isinstance(change["value"], np.ndarray)
            and np.allclose(change["previous"], change["value"])
        ):
            return
        for mat in self._clear_on_sigma_update:
            if hasattr(self, mat):
                delattr(self, mat)

    @properties.observer("rho")
    def _clear_sigma_mats_on_rho_update(self, change):
        if change["previous"] is change["value"]:
            return
        if (
            isinstance(change["previous"], np.ndarray)
            and isinstance(change["value"], np.ndarray)
            and np.allclose(change["previous"], change["value"])
        ):
            return
        for mat in self._clear_on_sigma_update:
            if hasattr(self, mat):
                delattr(self, mat)

    @property
    def Me(self):
        """
            Edge inner product matrix
        """
        if getattr(self, "_Me", None) is None:
            self._Me = self.mesh.getEdgeInnerProduct()
        return self._Me

    @property
    def MeI(self):
        """
            Edge inner product matrix
        """
        if getattr(self, "_MeI", None) is None:
            self._MeI = self.mesh.getEdgeInnerProduct(invMat=True)
        return self._MeI

    @property
    def Mf(self):
        """
            Face inner product matrix
        """
        if getattr(self, "_Mf", None) is None:
            self._Mf = self.mesh.getFaceInnerProduct()
        return self._Mf

    @property
    def MfI(self):
        """
            Face inner product matrix
        """
        if getattr(self, "_MfI", None) is None:
            self._MfI = self.mesh.getFaceInnerProduct(invMat=True)
        return self._MfI

    @property
    def Vol(self):
        if getattr(self, "_Vol", None) is None:
            self._Vol = sdiag(self.mesh.vol)
        return self._Vol

    ####################################################
    # Magnetic Permeability
    ####################################################
    @property
    def MfMui(self):
        """
        Face inner product matrix for \\(\\mu^{-1}\\).
        Used in the E-B formulation
        """
        if getattr(self, "_MfMui", None) is None:
            self._MfMui = self.mesh.getFaceInnerProduct(self.mui)
        return self._MfMui

    def MfMuiDeriv(self, u, v=None, adjoint=False):
        """
        Derivative of :code:`MfMui` with respect to the model.
        """
        if self.muiMap is None:
            return Zero()

        if getattr(self, "_MfMuiDeriv", None) is None:
            self._MfMuiDeriv = (
                self.mesh.getFaceInnerProductDeriv(np.ones(self.mesh.nC))(
                    np.ones(self.mesh.nF)
                )
                * self.muiDeriv
            )

        if v is not None:
            if not isinstance(u, Zero):
                u = u.flatten()
                if v.ndim > 1:
                    u = u[:, None]
            if adjoint is True:
                return self._MfMuiDeriv.T * (u * v)
            return u * (self._MfMuiDeriv * v)
        else:
            mat = sdiag(u) * self._MfMuiDeriv
            if adjoint is True:
                return mat.T
            return mat

    @property
    def MfMuiI(self):
        """
        Inverse of :code:`MfMui`.
        """
        if getattr(self, "_MfMuiI", None) is None:
            self._MfMuiI = self.mesh.getFaceInnerProduct(self.mui, invMat=True)
        return self._MfMuiI

    def MfMuiIDeriv(self, u, v=None, adjoint=False):
        """
        Derivative of :code:`MfMui` with respect to the model
        """

        if self.muiMap is None:
            return Zero()

        if len(self.mui.shape) > 1:
            if self.mui.shape[1] > self.mesh.dim:
                raise NotImplementedError(
                    "Full anisotropy is not implemented for MfMuiIDeriv."
                )

        dMfMuiI_dI = -self.MfMuiI ** 2
        if adjoint is True:
            return self.MfMuiDeriv(
                u,
                v=dMfMuiI_dI.T * v if v is not None else dMfMuiI_dI.T,
                adjoint=adjoint,
            )
        return dMfMuiI_dI * self.MfMuiDeriv(u, v=v)

    @property
    def MeMu(self):
        """
        Edge inner product matrix for \\(\\mu\\).
        Used in the H-J formulation
        """
        if getattr(self, "_MeMu", None) is None:
            self._MeMu = self.mesh.getEdgeInnerProduct(self.mu)
        return self._MeMu

    def MeMuDeriv(self, u, v=None, adjoint=False):
        """
        Derivative of :code:`MeMu` with respect to the model.
        """
        if self.muMap is None:
            return Zero()

        if getattr(self, "_MeMuDeriv", None) is None:
            self._MeMuDeriv = (
                self.mesh.getEdgeInnerProductDeriv(np.ones(self.mesh.nC))(
                    np.ones(self.mesh.nE)
                )
                * self.muDeriv
            )

        if v is not None:
            if not isinstance(u, Zero):
                u = u.flatten()
                if v.ndim > 1:
                    # promote u iff v is a matrix
                    u = u[:, None]  # Avoids constructing the sparse matrix
            if adjoint:
                return self._MeMuDeriv.T * (u * v)
            return u * (self._MeMuDeriv * v)
        else:
            mat = sdiag(u) * self._MeMuDeriv
            if adjoint is True:
                return mat.T
            return mat

    @property
    def MeMuI(self):
        """
        Inverse of :code:`MeMu`
        """
        if getattr(self, "_MeMuI", None) is None:
            self._MeMuI = self.mesh.getEdgeInnerProduct(self.mu, invMat=True)
        return self._MeMuI

    def MeMuIDeriv(self, u, v=None, adjoint=False):
        """
        Derivative of :code:`MeMuI` with respect to the model
        """

        if self.muMap is None:
            return Zero()

        if len(self.mu.shape) > 1:
            if self.mu.shape[1] > self.mesh.dim:
                raise NotImplementedError(
                    "Full anisotropy is not implemented for MeMuIDeriv."
                )

        dMeMuI_dI = -self.MeMuI ** 2
        if adjoint is True:
            return self.MeMuDeriv(
                u, v=dMeMuI_dI.T * v if v is not None else dMeMuI_dI.T, adjoint=adjoint
            )
        return dMeMuI_dI * self.MeMuDeriv(u, v=v)

    ####################################################
    # Electrical Conductivity
    ####################################################
    @property
    def MeSigma(self):
        """
        Edge inner product matrix for \\(\\sigma\\).
        Used in the E-B formulation
        """
        if getattr(self, "_MeSigma", None) is None:
            self._MeSigma = self.mesh.getEdgeInnerProduct(self.sigma)
        return self._MeSigma

    def MeSigmaDeriv(self, u, v=None, adjoint=False):
        """
        Derivative of MeSigma with respect to the model times a vector (u)
        """
        if self.sigmaMap is None:
            return Zero()

        if getattr(self, "_MeSigmaDeriv", None) is None:
            self._MeSigmaDeriv = (
                self.mesh.getEdgeInnerProductDeriv(np.ones(self.mesh.nC))(
                    np.ones(self.mesh.nE)
                )
                * self.sigmaDeriv
            )

        if v is not None:
            if not isinstance(u, Zero):
                u = u.flatten()  # u is either nUx1 or nU
                if v.ndim > 1:
                    # promote u iff v is a matrix
                    u = u[:, None]  # Avoids constructing the sparse matrix
            if adjoint:
                return self._MeSigmaDeriv.T * (u * v)
            return u * (self._MeSigmaDeriv * v)
        else:
            if adjoint is True:
                return self._MeSigmaDeriv.T * sdiag(u)
            return sdiag(u) * self._MeSigmaDeriv

    @property
    def MeSigmaI(self):
        """
        Inverse of the edge inner product matrix for \\(\\sigma\\).
        """
        if getattr(self, "_MeSigmaI", None) is None:
            self._MeSigmaI = self.mesh.getEdgeInnerProduct(self.sigma, invMat=True)
        return self._MeSigmaI

    def MeSigmaIDeriv(self, u, v=None, adjoint=False):
        """
        Derivative of :code:`MeSigmaI` with respect to the model
        """
        if self.sigmaMap is None:
            return Zero()

        if len(self.sigma.shape) > 1:
            if self.sigma.shape[1] > self.mesh.dim:
                raise NotImplementedError(
                    "Full anisotropy is not implemented for MeSigmaIDeriv."
                )
        dMeSigmaI_dI = -self.MeSigmaI ** 2
        if adjoint is True:
            return self.MeSigmaDeriv(
                u,
                v=(dMeSigmaI_dI.T * v) if v is not None else dMeSigmaI_dI.T,
                adjoint=adjoint,
            )
        else:
            return dMeSigmaI_dI * self.MeSigmaDeriv(u, v=v)

    @property
    def MfRho(self):
        """
        Face inner product matrix for \\(\\rho\\). Used in the H-J
        formulation
        """
        if getattr(self, "_MfRho", None) is None:
            self._MfRho = self.mesh.getFaceInnerProduct(self.rho)
        return self._MfRho

    def MfRhoDeriv(self, u, v=None, adjoint=False):
        """
        Derivative of :code:`MfRho` with respect to the model.
        """
        if self.rhoMap is None:
            return Zero()

        if getattr(self, "_MfRhoDeriv", None) is None:
            self._MfRhoDeriv = (
                self.mesh.getFaceInnerProductDeriv(np.ones(self.mesh.nC))(
                    np.ones(self.mesh.nF)
                )
                * self.rhoDeriv
            )

        if v is not None:
            if not isinstance(u, Zero):
                u = u.flatten()
                if v.ndim > 1:
                    # promote u iff v is a matrix
                    u = u[:, None]  # Avoids constructing the sparse matrix
            if adjoint is True:
                return self._MfRhoDeriv.T.dot(u * v)
            return u * (self._MfRhoDeriv.dot(v))
        else:
            if adjoint is True:
                return self._MfRhoDeriv.T.dot(sdiag(u))
            return sdiag(u) * (self._MfRhoDeriv)

    @property
    def MfRhoI(self):
        """
        Inverse of :code:`MfRho`
        """
        if getattr(self, "_MfRhoI", None) is None:
            self._MfRhoI = self.mesh.getFaceInnerProduct(self.rho, invMat=True)
        return self._MfRhoI

    def MfRhoIDeriv(self, u, v=None, adjoint=False):
        """
            Derivative of :code:`MfRhoI` with respect to the model.
        """
        if self.rhoMap is None:
            return Zero()

        if len(self.rho.shape) > 1:
            if self.rho.shape[1] > self.mesh.dim:
                raise NotImplementedError(
                    "Full anisotropy is not implemented for MfRhoIDeriv."
                )
        dMfRhoI_dI = -self.MfRhoI ** 2

        if adjoint is True:
            return self.MfRhoDeriv(dMfRhoI_dI.T.dot(u), v=v, adjoint=adjoint)
        else:
            return dMfRhoI_dI.dot(self.MfRhoDeriv(u, v=v))


###############################################################################
#                                                                             #
#                             Base EM Source                                  #
#                                                                             #
###############################################################################


class BaseEMSrc(BaseSrc):
    """
    Base class for an EM sources
    """

    integrate = properties.Bool("integrate the source term?", default=False)

    def eval(self, simulation):
        """
        - :math:`s_m` : magnetic source term
        - :math:`s_e` : electric source term

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :rtype: tuple
        :return: tuple with magnetic source term and electric source term
        """
        s_m = self.s_m(simulation)
        s_e = self.s_e(simulation)
        return s_m, s_e

    def evalDeriv(self, simulation, v=None, adjoint=False):
        """
        Derivatives of the source terms with respect to the inversion model
        - :code:`s_mDeriv` : derivative of the magnetic source term
        - :code:`s_eDeriv` : derivative of the electric source term

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: tuple
        :return: tuple with magnetic source term and electric source term
            derivatives times a vector

        """
        if v is not None:
            return (
                self.s_mDeriv(simulation, v, adjoint),
                self.s_eDeriv(simulation, v, adjoint),
            )
        else:
            return (
                lambda v: self.s_mDeriv(simulation, v, adjoint),
                lambda v: self.s_eDeriv(simulation, v, adjoint),
            )

    def s_m(self, simulation):
        """
        Magnetic source term

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :rtype: numpy.ndarray
        :return: magnetic source term on mesh
        """
        return Zero()

    def s_e(self, simulation):
        """
        Electric source term

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :rtype: numpy.ndarray
        :return: electric source term on mesh
        """
        return Zero()

    def s_mDeriv(self, simulation, v, adjoint=False):
        """
        Derivative of magnetic source term with respect to the inversion model

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of magnetic source term derivative with a vector
        """

        return Zero()

    def s_eDeriv(self, simulation, v, adjoint=False):
        """
        Derivative of electric source term with respect to the inversion model

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of electric source term derivative with a vector
        """
        return Zero()
