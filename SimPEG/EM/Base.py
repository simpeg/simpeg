from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import properties
from scipy.constants import mu_0
import numpy as np

from SimPEG import Survey
from SimPEG import Problem
from SimPEG import Utils
from SimPEG import Maps
from SimPEG import Props
from SimPEG import Solver as SimpegSolver


__all__ = ['BaseEMProblem', 'BaseEMSurvey', 'BaseEMSrc']


###############################################################################
#                                                                             #
#                             Base EM Problem                                 #
#                                                                             #
###############################################################################

class BaseEMProblem(Problem.BaseProblem):

    sigma, sigmaMap, sigmaDeriv = Props.Invertible(
        "Electrical conductivity (S/m)"
    )

    rho, rhoMap, rhoDeriv = Props.Invertible(
        "Electrical resistivity (Ohm m)"
    )

    Props.Reciprocal(sigma, rho)

    mu = Props.PhysicalProperty(
        "Magnetic Permeability (H/m)",
        default=mu_0
    )
    mui = Props.PhysicalProperty(
        "Inverse Magnetic Permeability (m/H)"
    )

    Props.Reciprocal(mu, mui)

    surveyPair = Survey.BaseSurvey  #: The survey to pair with.
    dataPair = Survey.Data  #: The data to pair with.

    mapPair = Maps.IdentityMap  #: Type of mapping to pair with

    Solver = SimpegSolver  #: Type of solver to pair with
    solverOpts = {}  #: Solver options

    verbose = False

    ####################################################
    # Make A Symmetric
    ####################################################
    @property
    def _makeASymmetric(self):
        if getattr(self, '__makeASymmetric', None) is None:
            self.__makeASymmetric = True
        return self.__makeASymmetric

    ####################################################
    # Mass Matrices
    ####################################################
    @property
    def _clear_on_mu_update(self):
        return ['_MeMu', '_MeMuI', '_MfMui', '_MfMuiI']

    @property
    def _clear_on_sigma_update(self):
        return ['_MeSigma', '_MeSigmaI', '_MfRho', '_MfRhoI']

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.sigmaMap is not None or self.rhoMap is not None:
            toDelete += self._clear_on_sigma_update

        if hasattr(self, 'muMap') or hasattr(self, 'muiMap'):
            if self.muMap is not None or self.muiMap is not None:
                toDelete += self._clear_on_mu_update
        return toDelete

    @properties.observer('mu')
    def _clear_mu_mats_on_mu_update(self, change):
        if change['previous'] is change['value']:
            return
        if (
            isinstance(change['previous'], np.ndarray) and
            isinstance(change['value'], np.ndarray) and
            np.allclose(change['previous'], change['value'])
        ):
            return
        for mat in self._clear_on_mu_update:
            if hasattr(self, mat):
                delattr(self, mat)

    @properties.observer('mui')
    def _clear_mu_mats_on_mui_update(self, change):
        if change['previous'] is change['value']:
            return
        if (
            isinstance(change['previous'], np.ndarray) and
            isinstance(change['value'], np.ndarray) and
            np.allclose(change['previous'], change['value'])
        ):
            return
        for mat in self._clear_on_mu_update:
            if hasattr(self, mat):
                delattr(self, mat)

    @properties.observer('sigma')
    def _clear_sigma_mats_on_sigma_update(self, change):
        if change['previous'] is change['value']:
            return
        if (
            isinstance(change['previous'], np.ndarray) and
            isinstance(change['value'], np.ndarray) and
            np.allclose(change['previous'], change['value'])
        ):
            return
        for mat in self._clear_on_sigma_update:
            if hasattr(self, mat):
                delattr(self, mat)

    @properties.observer('rho')
    def _clear_sigma_mats_on_rho_update(self, change):
        if change['previous'] is change['value']:
            return
        if (
            isinstance(change['previous'], np.ndarray) and
            isinstance(change['value'], np.ndarray) and
            np.allclose(change['previous'], change['value'])
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
        if getattr(self, '_Me', None) is None:
            self._Me = self.mesh.getEdgeInnerProduct()
        return self._Me

    @property
    def MeI(self):
        """
            Edge inner product matrix
        """
        if getattr(self, '_MeI', None) is None:
            self._MeI = self.mesh.getEdgeInnerProduct(invMat=True)
        return self._MeI

    @property
    def Mf(self):
        """
            Face inner product matrix
        """
        if getattr(self, '_Mf', None) is None:
            self._Mf = self.mesh.getFaceInnerProduct()
        return self._Mf

    @property
    def MfI(self):
        """
            Face inner product matrix
        """
        if getattr(self, '_MfI', None) is None:
            self._MfI = self.mesh.getFaceInnerProduct(invMat=True)
        return self._MfI

    @property
    def Vol(self):
        if getattr(self, '_Vol', None) is None:
            self._Vol = Utils.sdiag(self.mesh.vol)
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
        if getattr(self, '_MfMui', None) is None:
            self._MfMui = self.mesh.getFaceInnerProduct(self.mui)
        return self._MfMui

    def MfMuiDeriv(self, u):
        """
        Derivative of :code:`MfMui` with respect to the model.
        """
        if self.muiMap is None:
            return Utils.Zero()

        return (
            self.mesh.getFaceInnerProductDeriv(self.mui)(u) * self.muiDeriv
        )

    @property
    def MfMuiI(self):
        """
        Inverse of :code:`MfMui`.
        """
        if getattr(self, '_MfMuiI', None) is None:
            self._MfMuiI = self.mesh.getFaceInnerProduct(self.mui, invMat=True)
        return self._MfMuiI

    # TODO: This should take a vector
    def MfMuiIDeriv(self, u):
        """
        Derivative of :code:`MfMui` with respect to the model
        """

        if self.muiMap is None:
            return Utils.Zero()

        if len(self.mui.shape) > 1:
            if self.mui.shape[1] > self.mesh.dim:
                raise NotImplementedError(
                        "Full anisotropy is not implemented for MfMuiIDeriv."
                )

        dMfMuiI_dI = -self.MfMuiI**2
        dMf_dmui = self.mesh.getEdgeInnerProductDeriv(self.mui)(u)
        return dMfMuiI_dI * (dMf_dmui * self.muiDeriv)


    @property
    def MeMu(self):
        """
        Edge inner product matrix for \\(\\mu\\).
        Used in the H-J formulation
        """
        if getattr(self, '_MeMu', None) is None:
            self._MeMu = self.mesh.getEdgeInnerProduct(self.mu)
        return self._MeMu

    def MeMuDeriv(self, u):
        """
        Derivative of :code:`MeMu` with respect to the model.
        """
        if self.muMap is None:
            return Utils.Zero()

        return (
            self.mesh.getEdgeInnerProductDeriv(self.mu)(u) * self.muDeriv
        )

    @property
    def MeMuI(self):
        """
            Inverse of :code:`MeMu`
        """
        if getattr(self, '_MeMuI', None) is None:
            self._MeMuI = self.mesh.getEdgeInnerProduct(self.mu, invMat=True)
        return self._MeMuI

    # TODO: This should take a vector
    def MeMuIDeriv(self, u):
        """
        Derivative of :code:`MeMuI` with respect to the model
        """

        if self.muMap is None:
            return Utils.Zero()

        if len(self.mu.shape) > 1:
            if self.mu.shape[1] > self.mesh.dim:
                raise NotImplementedError(
                    "Full anisotropy is not implemented for MeMuIDeriv."
                )

        dMeMuI_dI = -self.MeMuI**2
        dMe_dmu = self.mesh.getEdgeInnerProductDeriv(self.mu)(u)
        return dMeMuI_dI * (dMe_dmu * self.muDeriv)

    ####################################################
    # Electrical Conductivity
    ####################################################
    @property
    def MeSigma(self):
        """
        Edge inner product matrix for \\(\\sigma\\).
        Used in the E-B formulation
        """
        if getattr(self, '_MeSigma', None) is None:
            self._MeSigma = self.mesh.getEdgeInnerProduct(self.sigma)
        return self._MeSigma

    # TODO: This should take a vector
    def MeSigmaDeriv(self, u):
        """
        Derivative of MeSigma with respect to the model
        """
        if self.sigmaMap is None:
            return Utils.Zero()

        return (
            self.mesh.getEdgeInnerProductDeriv(self.sigma)(u) *
            self.sigmaDeriv
        )


    @property
    def MeSigmaI(self):
        """
        Inverse of the edge inner product matrix for \\(\\sigma\\).
        """
        if getattr(self, '_MeSigmaI', None) is None:
            self._MeSigmaI = self.mesh.getEdgeInnerProduct(
                self.sigma, invMat=True
            )
        return self._MeSigmaI

    # TODO: This should take a vector
    def MeSigmaIDeriv(self, u):
        """
        Derivative of :code:`MeSigmaI` with respect to the model
        """
        if self.sigmaMap is None:
            return Utils.Zero()

        if len(self.sigma.shape) > 1:
            if self.sigma.shape[1] > self.mesh.dim:
                raise NotImplementedError(
                    "Full anisotropy is not implemented for MeSigmaIDeriv."
                )

        dMeSigmaI_dI = -self.MeSigmaI**2
        dMe_dsig = self.mesh.getEdgeInnerProductDeriv(self.sigma)(u)
        return dMeSigmaI_dI * (dMe_dsig * self.sigmaDeriv)

    @property
    def MfRho(self):
        """
        Face inner product matrix for \\(\\rho\\). Used in the H-J
        formulation
        """
        if getattr(self, '_MfRho', None) is None:
            self._MfRho = self.mesh.getFaceInnerProduct(self.rho)
        return self._MfRho

    # TODO: This should take a vector
    def MfRhoDeriv(self, u):
        """
        Derivative of :code:`MfRho` with respect to the model.
        """
        if self.rhoMap is None:
            return Utils.Zero()

        return (
            self.mesh.getFaceInnerProductDeriv(self.rho)(u) * self.rhoDeriv
        )

    @property
    def MfRhoI(self):
        """
        Inverse of :code:`MfRho`
        """
        if getattr(self, '_MfRhoI', None) is None:
            self._MfRhoI = self.mesh.getFaceInnerProduct(self.rho, invMat=True)
        return self._MfRhoI

    # TODO: This should take a vector
    def MfRhoIDeriv(self, u):
        """
            Derivative of :code:`MfRhoI` with respect to the model.
        """
        if self.rhoMap is None:
            return Utils.Zero()

        if len(self.rho.shape) > 1:
            if self.rho.shape[1] > self.mesh.dim:
                raise NotImplementedError(
                    "Full anisotropy is not implemented for MfRhoIDeriv."
                )

        dMfRhoI_dI = -self.MfRhoI**2
        dMf_drho = self.mesh.getFaceInnerProductDeriv(self.rho)(u)
        return dMfRhoI_dI * (dMf_drho * self.rhoDeriv)


###############################################################################
#                                                                             #
#                             Base EM Survey                                  #
#                                                                             #
###############################################################################

class BaseEMSurvey(Survey.BaseSurvey):

    def __init__(self, srcList, **kwargs):
        # Sort these by frequency
        self.srcList = srcList
        Survey.BaseSurvey.__init__(self, **kwargs)

    def eval(self, f):
        """Project fields to receiver locations

        :param Fields u: fields object
        :rtype: numpy.ndarray
        :return: data
        """
        data = Survey.Data(self)
        for src in self.srcList:
            for rx in src.rxList:
                data[src, rx] = rx.eval(src, self.mesh, f)
        return data

    def evalDeriv(self, f):
        raise Exception('Use Receivers to project fields deriv.')


###############################################################################
#                                                                             #
#                             Base EM Source                                  #
#                                                                             #
###############################################################################

class BaseEMSrc(Survey.BaseSrc):

    integrate = properties.Bool("integrate the source term?", default=False)

    def eval(self, prob):
        """
        - :math:`s_m` : magnetic source term
        - :math:`s_e` : electric source term

        :param BaseFDEMProblem prob: FDEM Problem
        :rtype: tuple
        :return: tuple with magnetic source term and electric source term
        """
        s_m = self.s_m(prob)
        s_e = self.s_e(prob)
        return s_m, s_e

    def evalDeriv(self, prob, v=None, adjoint=False):
        """
        Derivatives of the source terms with respect to the inversion model
        - :code:`s_mDeriv` : derivative of the magnetic source term
        - :code:`s_eDeriv` : derivative of the electric source term

        :param BaseFDEMProblem prob: FDEM Problem
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: tuple
        :return: tuple with magnetic source term and electric source term
            derivatives times a vector

        """
        if v is not None:
            return (
                self.s_mDeriv(prob, v, adjoint),
                self.s_eDeriv(prob, v, adjoint)
            )
        else:
            return (
                lambda v: self.s_mDeriv(prob, v, adjoint),
                lambda v: self.s_eDeriv(prob, v, adjoint)
            )

    def s_m(self, prob):
        """
        Magnetic source term

        :param BaseFDEMProblem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: magnetic source term on mesh
        """
        return Utils.Zero()

    def s_e(self, prob):
        """
        Electric source term

        :param BaseFDEMProblem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: electric source term on mesh
        """
        return Utils.Zero()

    def s_mDeriv(self, prob, v, adjoint=False):
        """
        Derivative of magnetic source term with respect to the inversion model

        :param BaseFDEMProblem prob: FDEM Problem
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of magnetic source term derivative with a vector
        """

        return Utils.Zero()

    def s_eDeriv(self, prob, v, adjoint=False):
        """
        Derivative of electric source term with respect to the inversion model

        :param BaseFDEMProblem prob: FDEM Problem
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of electric source term derivative with a vector
        """
        return Utils.Zero()
