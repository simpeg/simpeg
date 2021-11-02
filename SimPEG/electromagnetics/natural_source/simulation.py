import time
import sys
import numpy as np
import properties
from discretize.utils import Zero
from scipy.constants import mu_0
from ...utils.code_utils import deprecate_class

from ...utils import mkvc, sdiag
from ..frequency_domain.simulation import BaseFDEMSimulation, Simulation3DElectricField
from ..utils import omega
from .survey import Data, Survey1D
from .fields import (
    Fields1DPrimarySecondary,
    Fields1DElectricField,
    Fields1DMagneticField,
)


class BaseNSEMSimulation(BaseFDEMSimulation):
    """
    Base class for all Natural source problems.
    """

    # fieldsPair = BaseNSEMFields

    # def __init__(self, mesh, **kwargs):
    #     super(BaseNSEMSimulation, self).__init__()
    #     BaseFDEMProblem.__init__(self, mesh, **kwargs)
    #     setKwargs(self, **kwargs)
    # # Set the default pairs of the problem
    # surveyPair = Survey
    # dataPair = Data

    # Notes:
    # Use the fields and devs methods from BaseFDEMProblem

    # NEED to clean up the Jvec and Jtvec to use Zero and Identities for None components.
    def Jvec(self, m, v, f=None):
        """
        Function to calculate the data sensitivities dD/dm times a vector.

        :param numpy.ndarray m: conductivity model (nP,)
        :param numpy.ndarray v: vector which we take sensitivity product with (nP,)
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM (optional) u: NSEM fields object, if not given it is calculated
        :rtype: numpy.ndarray
        :return: Jv (nData,) Data sensitivities wrt m
        """

        # Calculate the fields if not given as input
        if f is None:
            f = self.fields(m)
        # Set current model
        self.model = m
        # Initiate the Jv object
        Jv = Data(self.survey)

        # Loop all the frequenies
        for freq in self.survey.frequencies:
            # Get the system
            A = self.getA(freq)
            # Factor
            Ainv = self.solver(A, **self.solver_opts)

            for src in self.survey.get_sources_by_frequency(freq):
                u_src = f[src, :]

                dA_dm_v = self.getADeriv(
                    freq, u_src, v
                )  # Size: nE,2 (u_px,u_py) in the columns.
                dRHS_dm_v = self.getRHSDeriv(
                    freq, v
                )  # Size: nE,2 (u_px,u_py) in the columns.
                # Calculate du/dm*v
                du_dm_v = Ainv * (-dA_dm_v + dRHS_dm_v)
                # Calculate the projection derivatives
                for rx in src.receiver_list:
                    # Calculate dP/du*du/dm*v
                    Jv[src, rx] = rx.evalDeriv(
                        src, self.mesh, f, mkvc(du_dm_v)
                    )  # wrt uPDeriv_u(mkvc(du_dm))
            Ainv.clean()

        return mkvc(Jv)

    def Jtvec(self, m, v, f=None):
        """
        Function to calculate the transpose of the data sensitivities (dD/dm)^T times a vector.

        :param numpy.ndarray m: inversion model (nP,)
        :param numpy.ndarray v: vector which we take adjoint product with (nP,)
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM f (optional): NSEM fields object, if not given it is calculated
        :rtype: numpy.ndarray
        :return: Jtv (nP,) Data sensitivities wrt m
        """

        if f is None:
            f = self.fields(m)

        self.model = m

        # Ensure v is a data object.
        if not isinstance(v, Data):
            v = Data(self.survey, v)

        Jtv = np.zeros(m.size)

        for freq in self.survey.frequencies:
            AT = self.getA(freq).T

            ATinv = self.solver(AT, **self.solver_opts)

            for src in self.survey.get_sources_by_frequency(freq):
                # u_src needs to have both polarizations
                u_src = f[src, :]

                for rx in src.receiver_list:
                    # Get the adjoint evalDeriv
                    # PTv needs to be nE,2
                    df_duT, df_dmT = rx.evalDeriv(
                        src, self.mesh, f, v=v[src, rx], adjoint=True
                    )  # wrt f, need possibility wrt m

                    ATinvdf_duT = ATinv * df_duT

                    dA_dmT = self.getADeriv(freq, u_src, ATinvdf_duT, adjoint=True)
                    dRHS_dmT = self.getRHSDeriv(freq, ATinvdf_duT, adjoint=True)
                    du_dmT = -dA_dmT + dRHS_dmT

                    df_dmT = df_dmT + du_dmT

                    Jtv += np.array(df_dmT, dtype=complex).real

            # Clean the factorization, clear memory.
            ATinv.clean()
        return Jtv


###################################
# 1D problems
###################################


class Simulation1DPrimarySecondary(BaseNSEMSimulation):
    """
    A NSEM problem soving a e formulation and primary/secondary fields decomposion.

    By eliminating the magnetic flux density using

        .. math ::

            \mathbf{b} = \\frac{1}{i \omega}\\left(-\mathbf{C} \mathbf{e} \\right)


    we can write Maxwell's equations as a second order system in \\\(\\\mathbf{e}\\\) only:

    .. math ::
        \\left[ \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^e } \mathbf{C} + i \omega \mathbf{M_{\sigma}^f} \\right] \mathbf{e}_{s} = i \omega \mathbf{M_{\sigma_{s}}^f } \mathbf{e}_{p}

    which we solve for :math:`\\mathbf{e_s}`. The total field :math:`\mathbf{e} = \mathbf{e_p} + \mathbf{e_s}`.

    The primary field is estimated from a background model (commonly half space ).


    """

    # From FDEMproblem: Used to project the fields. Currently not used for NSEMproblem.
    _solutionType = "e_1dSolution"
    _formulation = "EF"
    fieldsPair = Fields1DPrimarySecondary

    # Initiate properties
    _sigmaPrimary = None

    def __init__(self, mesh, **kwargs):
        BaseNSEMSimulation.__init__(self, mesh, **kwargs)
        # self._sigmaPrimary = sigmaPrimary

    @property
    def MeMui(self):
        """
        Edge inner product matrix
        """
        if getattr(self, "_MeMui", None) is None:
            self._MeMui = self.mesh.getEdgeInnerProduct(1.0 / mu_0)
        return self._MeMui

    @property
    def MfSigma(self):
        """
        Edge inner product matrix
        """
        # if getattr(self, '_MfSigma', None) is None:
        self._MfSigma = self.mesh.getFaceInnerProduct(self.sigma)
        return self._MfSigma

    def MfSigmaDeriv(self, u):
        """
        Edge inner product matrix
        """
        # if getattr(self, '_MfSigmaDeriv', None) is None:
        # print('[info mfsigmad] !!!!!!!!!!! ', u[:, 0])
        self._MfSigmaDeriv = (
            self.mesh.getFaceInnerProductDeriv(self.sigma)(u[:, 0]) * self.sigmaDeriv
        )
        return self._MfSigmaDeriv

    @property
    def sigmaPrimary(self):
        """
        A background model, use for the calculation of the primary fields.

        """
        return self._sigmaPrimary

    @sigmaPrimary.setter
    def sigmaPrimary(self, val):
        # Note: TODO add logic for val, make sure it is the correct size.
        self._sigmaPrimary = val

    def getA(self, freq):
        """
        Function to get the A matrix.

        :param float freq: Frequency
        :rtype: scipy.sparse.csr_matrix
        :return: A
        """

        # Note: need to use the code above since in the 1D problem I want
        # e to live on Faces(nodes) and h on edges(cells). Might need to rethink this
        # Possible that _fieldType and _eqLocs can fix this
        MeMui = self.MeMui
        MfSigma = self.MfSigma
        C = self.mesh.nodalGrad
        # Make A
        A = C.T * MeMui * C + 1j * omega(freq) * MfSigma
        # Either return full or only the inner part of A
        return A

    def getADeriv(self, freq, u, v, adjoint=False):
        """
        The derivative of A wrt sigma
        """

        u_src = u["e_1dSolution"]
        dMfSigma_dm = self.MfSigmaDeriv(u_src)
        if adjoint:
            return 1j * omega(freq) * mkvc(dMfSigma_dm.T * v,)
        # Note: output has to be nN/nF, not nC/nE.
        # v should be nC
        return 1j * omega(freq) * mkvc(dMfSigma_dm * v,)

    def getRHS(self, freq):
        """
        Function to return the right hand side for the system.

        :param float freq: Frequency
        :rtype: numpy.ndarray
        :return: RHS for 1 polarizations, primary fields (nF, 1)
        """

        # Get sources for the frequncy(polarizations)
        Src = self.survey.get_sources_by_frequency(freq)[0]
        # Only select the yx polarization
        S_e = mkvc(Src.S_e(self)[:, 1], 2)
        return -1j * omega(freq) * S_e

    def getRHSDeriv(self, freq, v, adjoint=False):
        """
        The derivative of the RHS wrt sigma
        """

        Src = self.survey.get_sources_by_frequency(freq)[0]

        S_eDeriv = mkvc(Src.s_eDeriv_m(self, v, adjoint),)
        return -1j * omega(freq) * S_eDeriv

    def fields(self, m=None):
        """
        Function to calculate all the fields for the model m.

        :param numpy.ndarray m: Conductivity model (nC,)
        :rtype: SimPEG.electromagnetics.natural_source.fields.Fields1DPrimarySecondary
        :return: NSEM fields object containing the solution
        """
        # Set the current model
        if m is not None:
            self.model = m
        # Make the fields object
        F = self.fieldsPair(self)
        # Loop over the frequencies
        for freq in self.survey.frequencies:
            if self.verbose:
                startTime = time.time()
                print("Starting work for {:.3e}".format(freq))
                sys.stdout.flush()
            A = self.getA(freq)
            rhs = self.getRHS(freq)
            Ainv = self.solver(A, **self.solver_opts)
            e_s = Ainv * rhs

            # Store the fields
            Src = self.survey.get_sources_by_frequency(freq)[0]
            # NOTE: only store the e_solution(secondary), all other components calculated in the fields object
            F[Src, "e_1dSolution"] = e_s

            if self.verbose:
                print("Ran for {:f} seconds".format(time.time() - startTime))
                sys.stdout.flush()
        return F


class Simulation1DElectricField(BaseFDEMSimulation):
    """
    1D finite volume simulation for the natural source electromagnetic problem.

    This corresponds to the TE mode 2D simulation where the electric field is
    located at cell centers and the magnetic flux is on edges.

    We are solving the discrete version of

    .. math::

        -\partial E_y + i \omega B_x = 0

        \partial_z \mu^{-1} B_x - sigma E_y = 0

    with default boundary conditions that $E_y[z_max] = 1$ (a plane wave source at
    the top of the domain), and $\partial_z E_y[z_min] = 0$ (the derivative of the field
    vanishes at the bottom of the domain).

    When we discretize, we obtain:

    .. math::

        \mathbf{b} = \frac{-1}{i\omega} \left( \mathbf{M^f}^{-1} \left((\mathbf{D}^\top \mathbf{V} - \mathbf{B}) \mathbf{e}\right) - \mathbf{e^{BC}} \right)

        \left( \mathbf{D} \mathbf{M^f}_{\mu^{-1}} \mathbf{M^f}^{-1} \left(\mathbf{D^\top} \mathbf{V} - \mathbf{B}\right) + i\omega \mathbf{M^{cc}}_{\sigma}\mathbf{V} \right) \mathbf{e} = \mathbf{D} \mathbf{M^f}_{\mu^{-1}}\mathbf{e^{BC}}

    """

    _solutionType = "eSolution"
    _formulation = "HJ"  # magnetic component is on edges
    fieldsPair = Fields1DElectricField
    _clear_on_sigma_update = ["_MccSigma", "_MfMuI"]

    # Must be 1D survey object
    survey = properties.Instance("a Survey1D survey object", Survey1D, required=True)

    def __init__(self, mesh, **kwargs):

        if mesh.dim > 1:
            raise ValueError(
                f"The mesh must be a 1D mesh. The provided mesh has dimension {mesh.dim}"
            )

        super().__init__(mesh, **kwargs)

        # corresponds to a dirichlet boundary at the top (= 1)
        # and nuemann boundary (= 0) at the bottom

        # TODO: Pass these boundary parameters as simulation properties
        alphas = np.r_[0, 1]
        betas = np.r_[1, 0]
        gammas = np.r_[0, 1]
        B, bc = mesh.cell_gradient_robin_weak_form(
            alpha=alphas, beta=betas, gamma=gammas,
        )
        self._B = B
        self._bc = bc

    @property
    def MccSigma(self):
        """
        Cell centered inner product matrix for conductivity
        """
        if getattr(self, "_MccSigma", None) is None:
            self._MccSigma = sdiag(self.sigma)
        return self._MccSigma

    def MccSigmaDeriv(self, u, v=None, adjoint=False):
        """
        Derivative of MccSigma with respect to the model times a vector
        """
        if self.sigmaMap is None:
            return Zero()

        if v is not None:
            if not adjoint:
                return u * (self.sigmaDeriv * v)
            elif adjoint:
                return self.sigmaDeriv.T * (u * v)
        else:
            mat = sdiag(u) * self.sigmaDeriv
            if not adjoint:
                return mat
            elif adjoint:
                return mat.T

    @property
    def MfMuI(self):
        if getattr(self, "_MfMuI", None) is None:
            self._MfMuI = self.mesh.get_face_inner_product(self.mu, invert_matrix=True)
        return self._MfMuI

    # TODO MfMuIDeriv

    def getA(self, freq):
        """
        System matrix

        .. math::

        \mathbf{A} = \mathbf{V} \mathbf{D} \mathbf{M^f}_{\mu^{-1}} \mathbf{M^f}^{-1} \mathbf{D^\top} - i\omega \mathbf{M^{cc}}_{\sigma}

        """
        V = self.Vol
        D = V @ self.mesh.faceDiv
        B = self._B
        MfMuI = self.MfMuI
        MccSigma = self.MccSigma

        return D @ MfMuI @ (D.T - B) + 1j * omega(freq) * MccSigma @ V

    def getADeriv(self, freq, u, v, adjoint=False):
        """
        Derivative with respect to the conductivity model
        """

        return 1j * omega(freq) * self.Vol @ self.MccSigmaDeriv(u, v, adjoint)

    def getRHS(self, freq):
        """
        Right hand side constructed using Dirichlet boundary conditions
        """
        D = self.mesh.faceDiv
        MfMuI = self.MfMuI
        e_bc = self._bc

        return self.Vol @ (D @ (MfMuI @ e_bc))


class Simulation1DMagneticField(BaseFDEMSimulation):
    """
    1D finite volume simulation for the natural source electromagnetic problem.

    This corresponds to the TM mode 2D simulation where the magnetic field is
    located at faces (nodes) and the electric field is on edges (cell_centers).
    """

    _solutionType = "hSolution"
    _formulation = "HJ"
    fieldsPair = Fields1DMagneticField

    # Must be 1D survey object
    survey = properties.Instance("a Survey1D survey object", Survey1D, required=True)

    def __init__(self, mesh, **kwargs):
        if mesh.dim > 1:
            raise ValueError(
                f"The mesh must be a 1D mesh. The provided mesh has dimension {mesh.dim}"
            )

        super().__init__(mesh, **kwargs)

        # corresponds to a dirichlet boundaries at the top (= 1 ) and bottom(=0)
        # for the y component of electric field
        self._rhs = -mesh.boundary_node_vector_integral * [0 + 0j, 1 + 0j]

    def getA(self, freq):
        """
        system matrix
        """
        G = self.mesh.nodal_gradient
        MeRho = self.MeRho
        MnMu = self.MnMu

        return G.T.tocsr() @ MeRho @ G + 1j * omega(freq) * MnMu

    def getADeriv_rho(self, freq, u, v, adjoint=False):
        G = self.mesh.nodal_gradient

        if adjoint:
            return self.MeRhoDeriv(G * u, G * v, adjoint)
        return G.T * self.MeRhoDeriv(G * u, v, adjoint)

    def getADeriv_mu(self, freq, u, v, adjoint=False):
        MnMuDeriv = self.MnMuDeriv(u)

        if adjoint is True:
            return 1j * omega(freq) * (MnMuDeriv.T * v)

        return 1j * omega(freq) * (MnMuDeriv * v)

    def getRHS(self, freq):
        """
        right hand side
        """
        return self._rhs

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        return Zero()

    def getADeriv(self, freq, u, v, adjoint=False):
        return self.getADeriv_rho(freq, u, v, adjoint) + self.getADeriv_mu(
            freq, u, v, adjoint
        )


###################################
# 2D problems
###################################

# class Simulation2DElectricField(BaseFDEMSimulation):
#     """
#     A
#     """


###################################
# 3D problems
###################################


class Simulation3DPrimarySecondary(Simulation3DElectricField):
    """
    A NSEM problem solving a e formulation and a primary/secondary fields decompostion.

    By eliminating the magnetic flux density using

        .. math ::

            \mathbf{b} = \\frac{1}{i \omega}\\left(-\mathbf{C} \mathbf{e} \\right)


    we can write Maxwell's equations as a second order system in :math:`\mathbf{e}` only:

    .. math ::

        \\left[\mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{C} + i \omega \mathbf{M_{\sigma}^e} \\right] \mathbf{e}_{s} = i \omega \mathbf{M_{\sigma_{p}}^e} \mathbf{e}_{p}

    which we solve for :math:`\mathbf{e_s}`. The total field :math:`\mathbf{e} = \mathbf{e_p} + \mathbf{e_s}`.

    The primary field is estimated from a background model (commonly as a 1D model).

    """

    # Initiate properties
    _sigmaPrimary = None

    # fieldsPair = Fields3DPrimarySecondary

    @property
    def sigmaPrimary(self):
        """
        A background model, use for the calculation of the primary fields.

        """
        return self._sigmaPrimary

    @sigmaPrimary.setter
    def sigmaPrimary(self, val):
        # Note: TODO add logic for val, make sure it is the correct size.
        self._sigmaPrimary = val


############
# Deprecated
############


@deprecate_class(removal_version="0.16.0", future_warn=True)
class Problem3D_ePrimSec(Simulation3DPrimarySecondary):
    pass


@deprecate_class(removal_version="0.16.0", future_warn=True)
class Problem1D_ePrimSec(Simulation1DPrimarySecondary):
    pass
