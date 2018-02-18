from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import properties
from scipy.constants import mu_0

from .. import NewSurvey as Survey
from .. import Simulation
from .. import Utils


###############################################################################
#                                                                             #
#                             Base EM Source                                  #
#                                                                             #
###############################################################################

class BaseEMSrc(Survey.BaseSrc):
    """
    Base class for an electromagnetic source. All EM sources are defined by an
    electric source term :code:`s_e` associated with `Ampere's Law <https://em.geosci.xyz/content/maxwell1_fundamentals/formative_laws/ampere_maxwell.html>`_
    and a magnetic source term :code:`s_m` associated with `Faraday's Law <https://em.geosci.xyz/content/maxwell1_fundamentals/formative_laws/faraday.html>`_

    In the time domain:

    .. math::

        \nabla \times \vec{e} + \frac{\partial \vec{b}}{\partial t} = \vec{s}_m \\
        \nabla \times \vec{h} - \vec{j} = \vec{s}_e

    and in the frequency domain

    .. math::

        \nabla \times \vec{E} + i\omega\vec{B} = \vec{S}_m \\
        \nabla \times \vec{H} - \vec{J} = \vec{S}_e

    """

    integrate = properties.Bool("integrate the source term?", default=False)

    def eval(self, simulation):
        """
        - :math:`s_m` : magnetic source term
        - :math:`s_e` : electric source term

        :param BaseEMSimulation simulation: EM Simulation
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

        :param BaseEMSimulation simulation: EM Simulation
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: tuple
        :return: tuple with magnetic source term and electric source term
            derivatives times a vector

        """
        if v is not None:
            return (
                self.s_mDeriv(simulation, v, adjoint),
                self.s_eDeriv(simulation, v, adjoint)
            )
        else:
            return (
                lambda v: self.s_mDeriv(simulation, v, adjoint),
                lambda v: self.s_eDeriv(simulation, v, adjoint)
            )

    def s_m(self, simulation):
        """
        Magnetic source term

        :param BaseEMSimulation simulation: EM Simulation
        :rtype: numpy.ndarray
        :return: magnetic source term on mesh
        """
        return Utils.Zero()

    def s_e(self, simulation):
        """
        Electric source term

        :param BaseEMSimulation simulation: EM Simulation
        :rtype: numpy.ndarray
        :return: electric source term on mesh
        """
        return Utils.Zero()

    def s_mDeriv(self, simulation, v, adjoint=False):
        """
        Derivative of magnetic source term with respect to the inversion model

        :param BaseEMSimulation simulation: EM Simulation
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of magnetic source term derivative with a vector
        """

        return Utils.Zero()

    def s_eDeriv(self, simulation, v, adjoint=False):
        """
        Derivative of electric source term with respect to the inversion model

        :param BaseEMSimulation simulation: EM Simulation
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of electric source term derivative with a vector
        """
        return Utils.Zero()


# ###############################################################################
# #                                                                             #
# #                             Base EM Survey                                  #
# #                                                                             #
# ###############################################################################

# class BaseEMSurvey(Survey.BaseSurvey):
#     """
#     Base class for an electromagnetic survey.
#     """

#     srcList = properties.List(
#         "A list of sources for the survey",
#         properties.Instance(
#             "A SimPEG source",
#             BaseEMSrc
#         ),
#         required=True
#     )

#     def __init__(self, **kwargs):
#         Survey.BaseSurvey.__init__(self, **kwargs)


###############################################################################
#                                                                             #
#                             Base EM Simulation                                 #
#                                                                             #
###############################################################################

class BaseEMSimulation(Simulation.BaseSimulation):
    """
    Base class for an electromagnetic simulation
    """

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

    survey = properties.Instance(
        "a list of sources",
        BaseEMSurvey,
        required=True
    )

    _makeASymmetric = properties.Bool(
        "symmetrize the system matrix?",
        default=True
    )

    def __init__(self, **kwargs):
        super(BaseEMSimulation, self).__init__(**kwargs)

    ####################################################
    # Mass Matrices
    ####################################################
    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.sigmaMap is not None or self.rhoMap is not None:
            toDelete += ['_MeSigma', '_MeSigmaI', '_MfRho', '_MfRhoI']

        if hasattr(self, 'muMap') or hasattr(self, 'muiMap'):
            if self.muMap is not None or self.muiMap is not None:
                toDelete += ['_MeMu', '_MeMuI', '_MfMui', '_MfMuiI']
        return toDelete

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


