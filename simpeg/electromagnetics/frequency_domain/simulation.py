import numpy as np
import scipy.sparse as sp
from discretize.utils import Zero

from ... import props
from ...data import Data
from ...utils import mkvc, validate_type
from ..base import BaseEMSimulation
from ..utils import omega
from .survey import Survey
from .fields import (
    FieldsFDEM,
    Fields3DElectricField,
    Fields3DMagneticFluxDensity,
    Fields3DMagneticField,
    Fields3DCurrentDensity,
)

import warnings


class BaseFDEMSimulation(BaseEMSimulation):
    r"""Base finite volume FDEM simulation class.

    This class is used to define properties and methods necessary for solving
    3D frequency-domain EM problems. For a :math:`+i\omega t` Fourier convention,
    Maxwell's equations are expressed as:

    .. math::
        \begin{align}
        \nabla \times \vec{E} + i\omega \vec{B} &= - i \omega \vec{S}_m \\
        \nabla \times \vec{H} - \vec{J} &= \vec{S}_e
        \end{align}

    where the constitutive relations between fields and fluxes are given by:

    * :math:`\vec{J} = \sigma \vec{E}`
    * :math:`\vec{B} = \mu \vec{H}`

    and:

    * :math:`\vec{S}_m` represents a magnetic source term
    * :math:`\vec{S}_e` represents a current source term

    Child classes of ``BaseFDEMSimulation`` solve the above expression numerically
    for various cases using mimetic finite volume.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        The mesh.
    survey : .frequency_domain.survey.Survey
        The frequency-domain EM survey.
    forward_only : bool, optional
        If ``True``, the factorization for the inverse of the system matrix at each
        frequency is discarded after the fields are computed at that frequency.
        If ``False``, the factorizations of the system matrices for all frequencies are stored.
    permittivity : (n_cells,) numpy.ndarray, optional
        Dielectric permittivity (F/m) defined on the entire mesh. If ``None``, electric displacement
        is ignored. Please note that `permittivity` is not an invertible property, and that future
        development will result in the deprecation of this propery.
    storeJ : bool, optional
        Whether to compute and store the sensitivity matrix.
    """

    fieldsPair = FieldsFDEM
    permittivity = props.PhysicalProperty("Dielectric permittivity (F/m)")

    def __init__(
        self,
        mesh,
        survey=None,
        forward_only=False,
        permittivity=None,
        storeJ=False,
        **kwargs,
    ):
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self.forward_only = forward_only
        if permittivity is not None:
            warnings.warn(
                "Simulations using permittivity have not yet been thoroughly tested and derivatives are not implemented. Contributions welcome!",
                stacklevel=2,
            )
        self.permittivity = permittivity
        self.storeJ = storeJ

    @property
    def survey(self):
        """The FDEM survey object.

        Returns
        -------
        .frequency_domain.survey.Survey
            The FDEM survey object.
        """
        if self._survey is None:
            raise AttributeError("Simulation must have a survey set")
        return self._survey

    @survey.setter
    def survey(self, value):
        if value is not None:
            value = validate_type("survey", value, Survey, cast=False)
        self._survey = value
        self._survey = value

    @property
    def storeJ(self):
        """Whether to compute and store the sensitivity matrix.

        Returns
        -------
        bool
            Whether to compute and store the sensitivity matrix.
        """
        return self._storeJ

    @storeJ.setter
    def storeJ(self, value):
        self._storeJ = validate_type("storeJ", value, bool)

    @property
    def forward_only(self):
        """Whether to store the factorizations of the inverses of the system matrices.

        If ``True``, the factorization for the inverse of the system matrix at each
        frequency is discarded after the fields are computed at that frequency.
        If ``False``, the factorizations of the system matrices for all frequencies are stored.

        Returns
        -------
        bool
            Whether to store the factorizations of the inverses of the system matrices.
        """
        return self._forward_only

    @forward_only.setter
    def forward_only(self, value):
        self._forward_only = validate_type("forward_only", value, bool)

    def _get_admittivity(self, freq):
        if self.permittivity is not None:
            return self.sigma + 1j * self.permittivity * omega(freq)
        else:
            return self.sigma

    def _get_face_admittivity_property_matrix(
        self, freq, invert_model=False, invert_matrix=False
    ):
        """
        Face inner product matrix with permittivity and resistivity
        """
        yhat = self._get_admittivity(freq)
        return self.mesh.get_face_inner_product(
            yhat, invert_model=invert_model, invert_matrix=invert_matrix
        )

    def _get_edge_admittivity_property_matrix(
        self, freq, invert_model=False, invert_matrix=False
    ):
        """
        Face inner product matrix with permittivity and resistivity
        """
        yhat = self._get_admittivity(freq)
        return self.mesh.get_edge_inner_product(
            yhat, invert_model=invert_model, invert_matrix=invert_matrix
        )

    # @profile
    def fields(self, m=None):
        """Compute and return the fields for the model provided.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model.

        Returns
        -------
        .frequency_domain.fields.FieldsFDEM
            The FDEM fields object.
        """

        if m is not None:
            self.model = m

        try:
            self.Ainv
        except AttributeError:
            self.Ainv = len(self.survey.frequencies) * [None]

        f = self.fieldsPair(self)

        for i_f, freq in enumerate(self.survey.frequencies):
            A = self.getA(freq)
            rhs = self.getRHS(freq)
            Ainv = self.solver(A, **self.solver_opts)
            u = Ainv * rhs
            if not self.forward_only:
                self.Ainv[i_f] = Ainv

            Srcs = self.survey.get_sources_by_frequency(freq)
            f[Srcs, self._solutionType] = u
        return f

    # @profile
    def Jvec(self, m, v, f=None):
        r"""Compute the sensitivity matrix times a vector.

        Where :math:`\mathbf{d}` are the data, :math:`\mathbf{m}` are the model parameters,
        and the sensitivity matrix is defined as:

        .. math::
            \mathbf{J} = \dfrac{\partial \mathbf{d}}{\partial \mathbf{m}}

        this method computes and returns the matrix-vector product:

        .. math::
            \mathbf{J v}

        for a given vector :math:`v`.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.
        v : (n_param,) numpy.ndarray
            The vector.
        f : .frequency_domain.fields.FieldsFDEM, optional
            Fields solved for all sources.

        Returns
        -------
        (n_data,) numpy.ndarray
            The sensitivity matrix times a vector.
        """

        if f is None:
            f = self.fields(m)

        self.model = m

        Jv = Data(self.survey)

        for nf, freq in enumerate(self.survey.frequencies):
            for src in self.survey.get_sources_by_frequency(freq):
                u_src = f[src, self._solutionType]
                dA_dm_v = self.getADeriv(freq, u_src, v, adjoint=False)
                dRHS_dm_v = self.getRHSDeriv(freq, src, v)
                du_dm_v = self.Ainv[nf] * (-dA_dm_v + dRHS_dm_v)
                for rx in src.receiver_list:
                    Jv[src, rx] = rx.evalDeriv(src, self.mesh, f, du_dm_v=du_dm_v, v=v)

        return Jv.dobs

    def Jtvec(self, m, v, f=None):
        r"""Compute the adjoint sensitivity matrix times a vector.

        Where :math:`\mathbf{d}` are the data, :math:`\mathbf{m}` are the model parameters,
        and the sensitivity matrix is defined as:

        .. math::
            \mathbf{J} = \dfrac{\partial \mathbf{d}}{\partial \mathbf{m}}

        this method computes and returns the matrix-vector product:

        .. math::
            \mathbf{J^T v}

        for a given vector :math:`v`.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.
        v : (n_data,) numpy.ndarray
            The vector.
        f : .frequency_domain.fields.FieldsFDEM, optional
            Fields solved for all sources.

        Returns
        -------
        (n_param,) numpy.ndarray
            The adjoint sensitivity matrix times a vector.
        """

        if f is None:
            f = self.fields(m)

        self.model = m

        # Ensure v is a data object.
        if not isinstance(v, Data):
            v = Data(self.survey, v)

        Jtv = np.zeros(m.size)

        for nf, freq in enumerate(self.survey.frequencies):
            for src in self.survey.get_sources_by_frequency(freq):
                u_src = f[src, self._solutionType]
                df_duT_sum = 0
                df_dmT_sum = 0
                for rx in src.receiver_list:
                    df_duT, df_dmT = rx.evalDeriv(
                        src, self.mesh, f, v=v[src, rx], adjoint=True
                    )
                    if not isinstance(df_duT, Zero):
                        df_duT_sum += df_duT
                    if not isinstance(df_dmT, Zero):
                        df_dmT_sum += df_dmT

                ATinvdf_duT = self.Ainv[nf] * df_duT_sum

                dA_dmT = self.getADeriv(freq, u_src, ATinvdf_duT, adjoint=True)
                dRHS_dmT = self.getRHSDeriv(freq, src, ATinvdf_duT, adjoint=True)
                du_dmT = -dA_dmT + dRHS_dmT

                df_dmT_sum += du_dmT
                Jtv += np.real(df_dmT_sum)

        return mkvc(Jtv)

    def getJ(self, m, f=None):
        r"""Generate the full sensitivity matrix.

        This method generates and stores the full sensitivity matrix for the
        model provided. I.e.:

        .. math::
            \mathbf{J} = \dfrac{\partial \mathbf{d}}{\partial \mathbf{m}}

        where :math:`\mathbf{d}` are the data and :math:`\mathbf{m}` are the model parameters.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.
        f : .static.resistivity.fields.FieldsDC, optional
            Fields solved for all sources.

        Returns
        -------
        (n_data, n_param) numpy.ndarray
            The full sensitivity matrix.
        """
        self.model = m

        if getattr(self, "_Jmatrix", None) is None:
            if f is None:
                f = self.fields(m)

            Ainv = self.Ainv
            m_size = self.model.size

            Jmatrix = np.zeros((self.survey.nD, m_size))

            data = Data(self.survey)

            for A_i, freq in zip(Ainv, self.survey.frequencies):
                for src in self.survey.get_sources_by_frequency(freq):
                    u_src = f[src, self._solutionType]

                    for rx in src.receiver_list:
                        v = np.eye(rx.nD, dtype=float)

                        df_duT, df_dmT = rx.evalDeriv(
                            src, self.mesh, f, v=v, adjoint=True
                        )

                        df_duT = np.hstack([df_duT])
                        ATinvdf_duT = A_i * df_duT
                        dA_dmT = self.getADeriv(freq, u_src, ATinvdf_duT, adjoint=True)
                        dRHS_dmT = self.getRHSDeriv(
                            freq, src, ATinvdf_duT, adjoint=True
                        )
                        du_dmT = -dA_dmT

                        if not isinstance(dRHS_dmT, Zero):
                            du_dmT += dRHS_dmT
                        if not isinstance(df_dmT[0], Zero):
                            du_dmT += np.hstack(df_dmT)

                        block = np.array(du_dmT, dtype=complex).real.T
                        data_inds = data.index_dictionary[src][rx]
                        Jmatrix[data_inds] = block

            self._Jmatrix = Jmatrix

        return self._Jmatrix

    def getJtJdiag(self, m, W=None, f=None):
        r"""Return the diagonal of :math:`\mathbf{J^T J}`.

        Where :math:`\mathbf{d}` are the data and :math:`\mathbf{m}` are the model parameters,
        the sensitivity matrix :math:`\mathbf{J}` is defined as:

        .. math::
            \mathbf{J} = \dfrac{\partial \mathbf{d}}{\partial \mathbf{m}}

        This method returns the diagonals of :math:`\mathbf{J^T J}`. When the
        *W* input argument is used to include a diagonal weighting matrix
        :math:`\mathbf{W}`, this method returns the diagonal of
        :math:`\mathbf{W^T J^T J W}`.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.
        W : (n_param, n_param) scipy.sparse.csr_matrix
            A diagonal weighting matrix.
        f : .frequency_domain.fields.FieldsFDEM, optional
            Fields solved for all sources.

        Returns
        -------
        (n_param,) numpy.ndarray
            The diagonals.
        """
        self.model = m

        if getattr(self, "_gtgdiag", None) is None:
            J = self.getJ(m, f=f)

            if W is None:
                W = np.ones(J.shape[0])
            else:
                W = W.diagonal() ** 2

            diag = np.einsum("i,ij,ij->j", W, J, J)

            self._gtgdiag = diag

        return self._gtgdiag

    # @profile
    def getSourceTerm(self, freq):
        r"""Returns the discrete source terms for the frequency provided.

        This method computes and returns the discrete magnetic and electric source
        terms for all soundings at the frequency provided. The exact shape and
        implementation of the source terms when solving for the fields at each frequency
        is formulation dependent.

        For definitions of the discrete magnetic (:math:`\mathbf{s_m}`) and electric
        (:math:`\mathbf{s_e}`) source terms for each simulation, see the *Notes* sections
        of the docstrings for:

        * :class:`.frequency_domain.Simulation3DElectricField`
        * :class:`.frequency_domain.Simulation3DMagneticField`
        * :class:`.frequency_domain.Simulation3DCurrentDensity`
        * :class:`.frequency_domain.Simulation3DMagneticFluxDensity`

        Parameters
        ----------
        freq : float
            The frequency in Hz.

        Returns
        -------
        s_m : numpy.ndarray
            The magnetic sources terms. (n_faces, n_sources) for EB-formulations. (n_edges, n_sources) for HJ-formulations.
        s_e : numpy.ndarray
            The electric sources terms. (n_edges, n_sources) for EB-formulations. (n_faces, n_sources) for HJ-formulations.
        """
        Srcs = self.survey.get_sources_by_frequency(freq)
        n_fields = sum(src._fields_per_source for src in Srcs)
        if self._formulation == "EB":
            s_m = np.zeros((self.mesh.nF, n_fields), dtype=complex, order="F")
            s_e = np.zeros((self.mesh.nE, n_fields), dtype=complex, order="F")
        elif self._formulation == "HJ":
            s_m = np.zeros((self.mesh.nE, n_fields), dtype=complex, order="F")
            s_e = np.zeros((self.mesh.nF, n_fields), dtype=complex, order="F")

        i = 0
        for src in Srcs:
            ii = i + src._fields_per_source
            smi, sei = src.eval(self)
            if not isinstance(smi, Zero) and smi.ndim == 1:
                smi = smi[:, None]
            if not isinstance(sei, Zero) and sei.ndim == 1:
                sei = sei[:, None]
            s_m[:, i:ii] = s_m[:, i:ii] + smi
            s_e[:, i:ii] = s_e[:, i:ii] + sei
            i = ii
        return s_m, s_e

    @property
    def deleteTheseOnModelUpdate(self):
        """List of model-dependent attributes to clean upon model update.

        Some of the FDEM simulation's attributes are model-dependent. This property specifies
        the model-dependent attributes that much be cleared when the model is updated.

        Returns
        -------
        list of str
            List of the model-dependent attributes to clean upon model update.
        """
        toDelete = super().deleteTheseOnModelUpdate
        return toDelete + ["_Jmatrix", "_gtgdiag"]


###############################################################################
#                               E-B Formulation                               #
###############################################################################


class Simulation3DElectricField(BaseFDEMSimulation):
    r"""3D FDEM simulation in terms of the electric field.

    This simulation solves for the electric field at each frequency.
    In this formulation, the electric fields are defined on mesh edges and the
    magnetic flux density is defined on mesh faces; i.e. it is an EB formulation.
    See the *Notes* section for a comprehensive description of the formulation.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        The mesh.
    survey : .frequency_domain.survey.Survey
        The frequency-domain EM survey.
    forward_only : bool, optional
        If ``True``, the factorization for the inverse of the system matrix at each
        frequency is discarded after the fields are computed at that frequency.
        If ``False``, the factorizations of the system matrices for all frequencies are stored.
    permittivity : (n_cells,) numpy.ndarray, optional
        Dielectric permittivity (F/m) defined on the entire mesh. If ``None``, electric displacement
        is ignored. Please note that `permittivity` is not an invertible property, and that future
        development will result in the deprecation of this propery.
    storeJ : bool, optional
        Whether to compute and store the sensitivity matrix.

    Notes
    -----
    Here, we start with the Maxwell's equations in the frequency-domain where a
    :math:`+i\omega t` Fourier convention is used:

    .. math::
        \begin{align}
        &\nabla \times \vec{E} + i\omega \vec{B} = - i \omega \vec{S}_m \\
        &\nabla \times \vec{H} - \vec{J} = \vec{S}_e
        \end{align}

    where :math:`\vec{S}_e` is an electric source term that defines a source current density,
    and :math:`\vec{S}_m` magnetic source term that defines a source magnetic flux density.
    We define the constitutive relations for the electrical conductivity :math:`\sigma`
    and magnetic permeability :math:`\mu` as:

    .. math::
        \vec{J} &= \sigma \vec{E} \\
        \vec{H} &= \mu^{-1} \vec{B}

    We then take the inner products of all previous expressions with a vector test function :math:`\vec{u}`.
    Through vector calculus identities and the divergence theorem, we obtain:

    .. math::
        & \int_\Omega \vec{u} \cdot (\nabla \times \vec{E}) \, dv
        + i \omega \int_\Omega \vec{u} \cdot \vec{B} \, dv
        = - i \omega \int_\Omega \vec{u} \cdot \vec{S}_m \, dv \\
        & \int_\Omega (\nabla \times \vec{u}) \cdot \vec{H} \, dv
        - \oint_{\partial \Omega} \vec{u} \cdot (\vec{H} \times \hat{n}) \, da
        - \int_\Omega \vec{u} \cdot \vec{J} \, dv
        = \int_\Omega \vec{u} \cdot \vec{S}_j \, dv \\
        & \int_\Omega \vec{u} \cdot \vec{J} \, dv = \int_\Omega \vec{u} \cdot \sigma \vec{E} \, dv \\
        & \int_\Omega \vec{u} \cdot \vec{H} \, dv = \int_\Omega \vec{u} \cdot \mu^{-1} \vec{B} \, dv

    Assuming natural boundary conditions, the surface integral is zero.

    The above expressions are discretized in space according to the finite volume method.
    The discrete electric fields :math:`\mathbf{e}` are defined on mesh edges,
    and the discrete magnetic flux densities :math:`\mathbf{b}` are defined on mesh faces.
    This implies :math:`\mathbf{j}` must be defined on mesh edges and :math:`\mathbf{h}` must
    be defined on mesh faces. Where :math:`\mathbf{u_e}` and :math:`\mathbf{u_f}` represent
    test functions discretized to edges and faces, respectively, we obtain the following
    set of discrete inner-products:

    .. math::
        &\mathbf{u_f^T M_f C e} + i \omega \mathbf{u_f^T M_f b} = - i \omega \mathbf{u_f^T M_f s_m} \\
        &\mathbf{u_e^T C^T M_f h} - \mathbf{u_e^T M_e j} = \mathbf{u_e^T s_e} \\
        &\mathbf{u_e^T M_e j} = \mathbf{u_e^T M_{e \sigma} e} \\
        &\mathbf{u_f^T M_f h} = \mathbf{u_f^T M_{f \mu} b}

    where

    * :math:`\mathbf{C}` is the discrete curl operator
    * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
    * :math:`\mathbf{M_e}` is the edge inner-product matrix
    * :math:`\mathbf{M_f}` is the face inner-product matrix
    * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities projected to edges
    * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inner-product matrix for inverse permeabilities projected to faces

    By cancelling like-terms and combining the discrete expressions to solve for the electric field, we obtain:

    .. math::
        \mathbf{A \, e} = \mathbf{q}

    where

    * :math:`\mathbf{A} = \mathbf{C^T M_{f\frac{1}{\mu}} C} + i\omega \mathbf{M_{e\sigma}}`
    * :math:`\mathbf{q} = - i \omega \mathbf{s_e} - i \omega \mathbf{C^T M_{f\frac{1}{\mu}} s_m }`

    """

    _solutionType = "eSolution"
    _formulation = "EB"
    fieldsPair = Fields3DElectricField

    def getA(self, freq):
        r"""System matrix for the frequency provided.

        This method returns the system matrix for the frequency provided:

        .. math::
            \mathbf{A} = \mathbf{C^T M_{f\frac{1}{\mu}} C} + i\omega \mathbf{M_{e\sigma}}

        where

        * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities projected to edges
        * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inner-product matrix for inverse permeabilities projected to faces

        See the *Notes* section of the doc strings for :class:`Simulation3DElectricField`
        for a full description of the formulation.

        Parameters
        ----------
        freq : float
            The frequency in Hz.

        Returns
        -------
        (n_edges, n_edges) sp.sparse.csr_matrix
            The system matrix.
        """

        MfMui = self.MfMui
        C = self.mesh.edge_curl

        if self.permittivity is None:
            MeSigma = self.MeSigma
            A = C.T.tocsr() * MfMui * C + 1j * omega(freq) * MeSigma
        else:
            Meyhat = self._get_edge_admittivity_property_matrix(freq)
            A = C.T.tocsr() * MfMui * C + 1j * omega(freq) * Meyhat

        return A

    def getADeriv_sigma(self, freq, u, v, adjoint=False):
        r"""Conductivity derivative operation for the system matrix times a vector.

        The system matrix at each frequency is given by:

        .. math::
            \mathbf{A} = \mathbf{C^T M_{f\frac{1}{\mu}} C} + i\omega \mathbf{M_{e\sigma}}

        where

        * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities projected to edges
        * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inner-product matrix for inverse permeabilities projected to faces

        See the *Notes* section of the doc strings for :class:`Simulation3DElectricField`
        for a full description of the formulation.

        Where :math:`\mathbf{m}_\boldsymbol{\sigma}` are the set of model parameters defining the conductivity,
        :math:`\mathbf{v}` is a vector and :math:`\mathbf{e}` is the discrete electric field solution, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A \, e})}{\partial \mathbf{m}_\boldsymbol{\sigma}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A \, e})}{\partial \mathbf{m}_\boldsymbol{\sigma}}^T \, \mathbf{v}

        Parameters
        ----------
        freq : float
            The frequency in Hz.
        u : (n_edges,) numpy.ndarray
            The solution for the fields for the current model at the specified frequency.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_edges,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_edges,) for the standard operation.
            (n_param,) for the adjoint operation.
        """

        dMe_dsig_v = self.MeSigmaDeriv(u, v, adjoint)
        return 1j * omega(freq) * dMe_dsig_v

    def getADeriv_mui(self, freq, u, v, adjoint=False):
        r"""Inverse permeability derivative operation for the system matrix times a vector.

        The system matrix at each frequency is given by:

        .. math::
            \mathbf{A} = \mathbf{C^T M_{f\frac{1}{\mu}} C} + i\omega \mathbf{M_{e\sigma}}

        where

        * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities projected to edges
        * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inner-product matrix for inverse permeabilities projected to faces

        See the *Notes* section of the doc strings for :class:`Simulation3DElectricField`
        for a full description of the formulation.

        Where :math:`\mathbf{m}_\boldsymbol{\mu}` are the set of model parameters defining the permeability,
        :math:`\mathbf{v}` is a vector and :math:`\mathbf{e}` is the discrete electric field solution, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A \, e})}{\partial \mathbf{m}_\boldsymbol{\mu}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A \, e})}{\partial \mathbf{m}_\boldsymbol{\mu}}^T \, \mathbf{v}

        Parameters
        ----------
        freq : float
            The frequency in Hz.
        u : (n_edges,) numpy.ndarray
            The solution for the fields for the current model at the specified frequency.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_edges,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_edges,) for the standard operation.
            (n_param,) for the adjoint operation.
        """

        C = self.mesh.edge_curl

        if adjoint:
            return self.MfMuiDeriv(C * u).T * (C * v)

        return C.T * (self.MfMuiDeriv(C * u) * v)

    def getADeriv(self, freq, u, v, adjoint=False):
        r"""Derivative operation for the system matrix times a vector.

        The system matrix at each frequency is given by:

        .. math::
            \mathbf{A} = \mathbf{C^T M_{f\frac{1}{\mu}} C} + i\omega \mathbf{M_{e\sigma}}

        where

        * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities projected to edges
        * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inner-product matrix for inverse permeabilities projected to faces

        See the *Notes* section of the doc strings for :class:`Simulation3DElectricField`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters defining the electromagnetic properties
        :math:`\mathbf{v}` is a vector and :math:`\mathbf{e}` is the discrete electric field solution, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A \, e})}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A \, e})}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        freq : float
            The frequency in Hz.
        u : (n_edges,) numpy.ndarray
            The solution for the fields for the current model at the specified frequency.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_edges,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_edges,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        return (
            self.getADeriv_sigma(freq, u, v, adjoint)
            + self.getADeriv_mui(freq, u, v, adjoint)
            # + self.getADeriv_permittivity(freq, u, v, adjoint)
        )

    def getRHS(self, freq):
        r"""Right-hand sides for the given frequency.

        This method returns the right-hand sides for the frequency provided.
        The right-hand side for each source is constructed according to:

        .. math::
            \mathbf{q} = - i \omega \mathbf{s_e} - i \omega \mathbf{C^T M_{f\frac{1}{\mu}} s_m }

        where

        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
        * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inner-product matrices for inverse permeabilities projected to faces

        See the *Notes* section of the doc strings for :class:`Simulation3DElectricField`
        for a full description of the formulation.

        Parameters
        ----------
        freq : float
            The frequency in Hz.

        Returns
        -------
        (n_edges, n_sources) numpy.ndarray
            The right-hand sides.
        """

        s_m, s_e = self.getSourceTerm(freq)
        C = self.mesh.edge_curl
        MfMui = self.MfMui

        return C.T * (MfMui * s_m) - 1j * omega(freq) * s_e

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        r"""Derivative of the right-hand side times a vector for a given source and frequency.

        The right-hand side for each source is constructed according to:

        .. math::
            \mathbf{q} = -i \omega \mathbf{s_e} - i \omega \mathbf{C^T M_{f\frac{1}{\mu}} s_m }

        where

        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
        * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inner-product matrices for inverse permeabilities projected to faces

        See the *Notes* section of the doc strings for :class:`Simulation3DElectricField`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters and :math:`\mathbf{v}` is a vector,
        this method returns

        .. math::
            \frac{\partial \mathbf{q_k}}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial \mathbf{q_k}}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        freq : int
            The frequency in Hz.
        src : .frequency_domain.sources.BaseFDEMSrc
            The FDEM source object.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_edges,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of the right-hand sides times a vector. (n_edges,) for the standard operation.
            (n_param,) for the adjoint operation.
        """

        C = self.mesh.edge_curl
        MfMui = self.MfMui
        s_m, s_e = self.getSourceTerm(freq)
        s_mDeriv, s_eDeriv = src.evalDeriv(self, adjoint=adjoint)
        MfMuiDeriv = self.MfMuiDeriv(s_m)

        if adjoint:
            return (
                s_mDeriv(MfMui * (C * v))
                + MfMuiDeriv.T * (C * v)
                - 1j * omega(freq) * s_eDeriv(v)
            )
        return C.T * (MfMui * s_mDeriv(v) + MfMuiDeriv * v) - 1j * omega(
            freq
        ) * s_eDeriv(v)


class Simulation3DMagneticFluxDensity(BaseFDEMSimulation):
    r"""3D FDEM simulation in terms of the magnetic flux field.

    This simulation solves for the magnetic flux density at each frequency.
    In this formulation, the electric fields are defined on mesh edges and the
    magnetic flux density is defined on mesh faces; i.e. it is an EB formulation.
    See the *Notes* section for a comprehensive description of the formulation.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        The mesh.
    survey : .frequency_domain.survey.Survey
        The frequency-domain EM survey.
    forward_only : bool, optional
        If ``True``, the factorization for the inverse of the system matrix at each
        frequency is discarded after the fields are computed at that frequency.
        If ``False``, the factorizations of the system matrices for all frequencies are stored.
    permittivity : (n_cells,) numpy.ndarray, optional
        Dielectric permittivity (F/m) defined on the entire mesh. If ``None``, electric displacement
        is ignored. Please note that `permittivity` is not an invertible property, and that future
        development will result in the deprecation of this propery.
    storeJ : bool, optional
        Whether to compute and store the sensitivity matrix.

    Notes
    -----
    Here, we start with the Maxwell's equations in the frequency-domain where a
    :math:`+i\omega t` Fourier convention is used:

    .. math::
        \begin{align}
        &\nabla \times \vec{E} + i\omega \vec{B} = - i \omega \vec{S}_m \\
        &\nabla \times \vec{H} - \vec{J} = \vec{S}_e
        \end{align}

    where :math:`\vec{S}_e` is an electric source term that defines a source current density,
    and :math:`\vec{S}_m` magnetic source term that defines a source magnetic flux density.
    We define the constitutive relations for the electrical conductivity :math:`\sigma`
    and magnetic permeability :math:`\mu` as:

    .. math::
        \vec{J} &= \sigma \vec{E} \\
        \vec{H} &= \mu^{-1} \vec{B}

    We then take the inner products of all previous expressions with a vector test function :math:`\vec{u}`.
    Through vector calculus identities and the divergence theorem, we obtain:

    .. math::
        & \int_\Omega \vec{u} \cdot (\nabla \times \vec{E}) \, dv
        + i \omega \int_\Omega \vec{u} \cdot \vec{B} \, dv
        = - i \omega \int_\Omega \vec{u} \cdot \vec{S}_m \, dv \\
        & \int_\Omega (\nabla \times \vec{u}) \cdot \vec{H} \, dv
        - \oint_{\partial \Omega} \vec{u} \cdot (\vec{H} \times \hat{n}) \, da
        - \int_\Omega \vec{u} \cdot \vec{J} \, dv
        = \int_\Omega \vec{u} \cdot \vec{S}_j \, dv \\
        & \int_\Omega \vec{u} \cdot \vec{J} \, dv = \int_\Omega \vec{u} \cdot \sigma \vec{E} \, dv \\
        & \int_\Omega \vec{u} \cdot \vec{H} \, dv = \int_\Omega \vec{u} \cdot \mu^{-1} \vec{B} \, dv

    Assuming natural boundary conditions, the surface integral is zero.

    The above expressions are discretized in space according to the finite volume method.
    The discrete electric fields :math:`\mathbf{e}` are defined on mesh edges,
    and the discrete magnetic flux densities :math:`\mathbf{b}` are defined on mesh faces.
    This implies :math:`\mathbf{j}` must be defined on mesh edges and :math:`\mathbf{h}` must
    be defined on mesh faces. Where :math:`\mathbf{u_e}` and :math:`\mathbf{u_f}` represent
    test functions discretized to edges and faces, respectively, we obtain the following
    set of discrete inner-products:

    .. math::
        &\mathbf{u_f^T M_f C e} + i \omega \mathbf{u_f^T M_f b} = - i \omega \mathbf{u_f^T M_f s_m} \\
        &\mathbf{u_e^T C^T M_f h} - \mathbf{u_e^T M_e j} = \mathbf{u_e^T s_e} \\
        &\mathbf{u_e^T M_e j} = \mathbf{u_e^T M_{e\sigma} e} \\
        &\mathbf{u_f^T M_f h} = \mathbf{u_f^T M_{f \mu} b}

    where

    * :math:`\mathbf{C}` is the discrete curl operator
    * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
    * :math:`\mathbf{M_e}` is the edge inner-product matrix
    * :math:`\mathbf{M_f}` is the face inner-product matrix
    * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities projected to edges
    * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inner-product matrix for inverse permeabilities projected to faces

    By cancelling like-terms and combining the discrete expressions to solve for the magnetic flux density, we obtain:

    .. math::
        \mathbf{A \, b} = \mathbf{q}

    where

    * :math:`\mathbf{A} = \mathbf{C M_{e\sigma}^{-1} C^T M_{f\frac{1}{\mu}}} + i\omega \mathbf{I}`
    * :math:`\mathbf{q} = \mathbf{C M_{e\sigma}^{-1} s_e} - i \omega \mathbf{s_m}`

    """

    _solutionType = "bSolution"
    _formulation = "EB"
    fieldsPair = Fields3DMagneticFluxDensity

    def getA(self, freq):
        r"""System matrix for the frequency provided.

        This method returns the system matrix for the frequency provided:

        .. math::
            \mathbf{A} = \mathbf{C M_{e\sigma}^{-1} C^T M_{f\frac{1}{\mu}}} + i\omega \mathbf{I}

        where

        * :math:`\mathbf{I}` is the identity matrix
        * :math:`\mathbf{C}` is the curl operator
        * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities projected to edges
        * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inner-product matrix for inverse permeabilities projected to faces

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticFluxDensity`
        for a full description of the formulation.

        Parameters
        ----------
        freq : float
            The frequency in Hz.

        Returns
        -------
        (n_faces, n_faces) sp.sparse.csr_matrix
            The system matrix.
        """

        MfMui = self.MfMui
        C = self.mesh.edge_curl
        iomega = 1j * omega(freq) * sp.eye(self.mesh.nF)

        if self.permittivity is None:
            MeSigmaI = self.MeSigmaI
            A = C * (MeSigmaI * (C.T.tocsr() * MfMui)) + iomega
        else:
            MeyhatI = self._get_edge_admittivity_property_matrix(
                freq, invert_matrix=True
            )
            A = C * (MeyhatI * (C.T.tocsr() * MfMui)) + iomega

        if self._makeASymmetric:
            return MfMui.T.tocsr() * A
        return A

    def getADeriv_sigma(self, freq, u, v, adjoint=False):
        r"""Conductivity derivative operation for the system matrix times a vector.

        The system matrix at each frequency is given by:

        .. math::
            \mathbf{A} = \mathbf{C M_{e\sigma}^{-1} C^T M_{f\frac{1}{\mu}}} + i\omega \mathbf{I}

        where

        * :math:`\mathbf{I}` is the identity matrix
        * :math:`\mathbf{C}` is the curl operator
        * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities projected to edges
        * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inner-product matrix for inverse permeabilities projected to faces

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticFluxDensity`
        for a full description of the formulation.

        Where :math:`\mathbf{m}_\boldsymbol{\sigma}` are the set of model parameters defining the conductivity,
        :math:`\mathbf{v}` is a vector and :math:`\mathbf{b}` is the discrete magnetic flux density solution,
        this method assumes the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A \, b})}{\partial \mathbf{m}_\boldsymbol{\sigma}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A \, b})}{\partial \mathbf{m}_\boldsymbol{\sigma}}^T \, \mathbf{v}

        Parameters
        ----------
        freq : float
            The frequency in Hz.
        u : (n_faces,) numpy.ndarray
            The solution for the fields for the current model at the specified frequency.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_faces,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_faces,) for the standard operation.
            (n_param,) for the adjoint operation.
        """

        MfMui = self.MfMui
        C = self.mesh.edge_curl
        MeSigmaIDeriv = self.MeSigmaIDeriv
        vec = C.T * (MfMui * u)

        if adjoint:
            return MeSigmaIDeriv(vec, C.T * v, adjoint)
        return C * MeSigmaIDeriv(vec, v, adjoint)

        # if adjoint:
        #     return MeSigmaIDeriv.T * (C.T * v)
        # return C * (MeSigmaIDeriv * v)

    def getADeriv_mui(self, freq, u, v, adjoint=False):
        r"""Inverse permeability derivative operation for the system matrix times a vector.

        The system matrix at each frequency is given by:

        .. math::
            \mathbf{A} = \mathbf{C M_{e\sigma}^{-1} C^T M_{f\frac{1}{\mu}}} + i\omega \mathbf{I}

        where

        * :math:`\mathbf{I}` is the identity matrix
        * :math:`\mathbf{C}` is the curl operator
        * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities projected to edges
        * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inner-product matrix for inverse permeabilities projected to faces

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticFluxDensity`
        for a full description of the formulation.

        Where :math:`\mathbf{m}_\boldsymbol{\mu}` are the set of model parameters defining the permeability,
        :math:`\mathbf{v}` is a vector and :math:`\mathbf{b}` is the discrete magnetic flux density solution,
        this method assumes the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A \, b})}{\partial \mathbf{m}_\boldsymbol{\mu}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A \, b})}{\partial \mathbf{m}_\boldsymbol{\mu}}^T \, \mathbf{v}

        Parameters
        ----------
        freq : float
            The frequency in Hz.
        u : (n_faces,) numpy.ndarray
            The solution for the fields for the current model at the specified frequency.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_faces,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_faces,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        MfMuiDeriv = self.MfMuiDeriv(u)
        MeSigmaI = self.MeSigmaI
        C = self.mesh.edge_curl

        if adjoint:
            return MfMuiDeriv.T * (C * (MeSigmaI.T * (C.T * v)))
        return C * (MeSigmaI * (C.T * (MfMuiDeriv * v)))

    def getADeriv(self, freq, u, v, adjoint=False):
        r"""Derivative operation for the system matrix times a vector.

        The system matrix at each frequency is given by:

        .. math::
            \mathbf{A} = \mathbf{C M_{e\sigma}^{-1} C^T M_{f\frac{1}{\mu}}} + i\omega \mathbf{I}

        where

        * :math:`\mathbf{I}` is the identity matrix
        * :math:`\mathbf{C}` is the curl operator
        * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities projected to edges
        * :math:`\mathbf{M_{f\frac{1}{\mu}}}` is the inner-product matrix for inverse permeabilities projected to faces

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticFluxDensity`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters defining the electromagnetic properties,
        :math:`\mathbf{v}` is a vector and :math:`\mathbf{b}` is the discrete magnetic flux density solution,
        this method assumes the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A \, b})}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A \, b})}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        freq : float
            The frequency in Hz.
        u : (n_faces,) numpy.ndarray
            The solution for the fields for the current model at the specified frequency.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_faces,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_faces,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        if adjoint is True and self._makeASymmetric:
            v = self.MfMui * v

        ADeriv = self.getADeriv_sigma(freq, u, v, adjoint) + self.getADeriv_mui(
            freq, u, v, adjoint
        )

        if adjoint is False and self._makeASymmetric:
            return self.MfMui.T * ADeriv

        return ADeriv

    def getRHS(self, freq):
        r"""Right-hand sides for the given frequency.

        This method returns the right-hand sides for the frequency provided.
        The right-hand side for each source is constructed according to:

        .. math::
            \mathbf{q} = \mathbf{C M_{e\sigma}^{-1} s_e} - i \omega \mathbf{s_m }

        where

        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
        * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities projected to edges

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticFluxDensity`
        for a full description of the formulation.

        Parameters
        ----------
        freq : float
            The frequency in Hz.

        Returns
        -------
        (n_faces, n_sources) numpy.ndarray
            The right-hand sides.
        """

        s_m, s_e = self.getSourceTerm(freq)
        C = self.mesh.edge_curl

        if self.permittivity is None:
            MeSigmaI = self.MeSigmaI
            RHS = s_m + C * (MeSigmaI * s_e)
        else:
            MeyhatI = self._get_edge_admittivity_property_matrix(
                freq, invert_matrix=True
            )
            RHS = s_m + C * (MeyhatI * s_e)

        if self._makeASymmetric is True:
            MfMui = self.MfMui
            return MfMui.T * RHS

        return RHS

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        r"""Derivative of the right-hand side times a vector for a given source and frequency.

        The right-hand side for each source is constructed according to:

        .. math::
            \mathbf{q} = \mathbf{C M_{e\sigma}^{-1} s_e} - i \omega \mathbf{s_m }

        where

        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
        * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities projected to edges

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticFluxDensity`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters and :math:`\mathbf{v}` is a vector,
        this method returns

        .. math::
            \frac{\partial \mathbf{q_k}}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial \mathbf{q_k}}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        freq : int
            The frequency in Hz.
        src : .frequency_domain.sources.BaseFDEMSrc
            The FDEM source object.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_faces,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of the right-hand sides times a vector. (n_faces,) for the standard operation.
            (n_param,) for the adjoint operation.
        """

        C = self.mesh.edge_curl
        s_m, s_e = src.eval(self)
        MfMui = self.MfMui

        if self._makeASymmetric and adjoint:
            v = self.MfMui * v

        # MeSigmaIDeriv = self.MeSigmaIDeriv(s_e)
        s_mDeriv, s_eDeriv = src.evalDeriv(self, adjoint=adjoint)

        if not adjoint:
            # RHSderiv = C * (MeSigmaIDeriv * v)
            RHSderiv = C * self.MeSigmaIDeriv(s_e, v, adjoint)
            SrcDeriv = s_mDeriv(v) + C * (self.MeSigmaI * s_eDeriv(v))
        elif adjoint:
            # RHSderiv = MeSigmaIDeriv.T * (C.T * v)
            RHSderiv = self.MeSigmaIDeriv(s_e, C.T * v, adjoint)
            SrcDeriv = s_mDeriv(v) + s_eDeriv(self.MeSigmaI.T * (C.T * v))

        if self._makeASymmetric is True and not adjoint:
            return MfMui.T * (SrcDeriv + RHSderiv)

        return RHSderiv + SrcDeriv


###############################################################################
#                               H-J Formulation                               #
###############################################################################


class Simulation3DCurrentDensity(BaseFDEMSimulation):
    r"""3D FDEM simulation in terms of the current density.

    This simulation solves for the current density at each frequency.
    In this formulation, the magnetic fields are defined on mesh edges and the
    current densities are defined on mesh faces; i.e. it is an HJ formulation.
    See the *Notes* section for a comprehensive description of the formulation.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        The mesh.
    survey : .frequency_domain.survey.Survey
        The frequency-domain EM survey.
    forward_only : bool, optional
        If ``True``, the factorization for the inverse of the system matrix at each
        frequency is discarded after the fields are computed at that frequency.
        If ``False``, the factorizations of the system matrices for all frequencies are stored.
    permittivity : (n_cells,) numpy.ndarray, optional
        Dielectric permittivity (F/m) defined on the entire mesh. If ``None``, electric displacement
        is ignored. Please note that `permittivity` is not an invertible property, and that future
        development will result in the deprecation of this propery.
    storeJ : bool, optional
        Whether to compute and store the sensitivity matrix.

    Notes
    -----
    Here, we start with the Maxwell's equations in the frequency-domain where a
    :math:`+i\omega t` Fourier convention is used:

    .. math::
        \begin{align}
        &\nabla \times \vec{E} + i\omega \vec{B} = - i \omega \vec{S}_m \\
        &\nabla \times \vec{H} - \vec{J} = \vec{S}_e
        \end{align}

    where :math:`\vec{S}_e` is an electric source term that defines a source current density,
    and :math:`\vec{S}_m` magnetic source term that defines a source magnetic flux density.
    For now, we neglect displacement current (the `permittivity` attribute is ``None``).
    We define the constitutive relations for the electrical resistivity :math:`\rho`
    and magnetic permeability :math:`\mu` as:

    .. math::
        \vec{E} &= \rho \vec{J} \\
        \vec{B} &= \mu \vec{H}

    We then take the inner products of all previous expressions with a vector test function :math:`\vec{u}`.
    Through vector calculus identities and the divergence theorem, we obtain:

    .. math::
        & \int_\Omega (\nabla \times \vec{u}) \cdot \vec{E} \; dv
        - \oint_{\partial \Omega} \vec{u} \cdot (\vec{E} \times \hat{n} ) \, da
        + i \omega \int_\Omega \vec{u} \cdot \vec{B} \, dv
        = - i \omega \int_\Omega \vec{u} \cdot \vec{S}_m dv \\
        & \int_\Omega \vec{u} \cdot (\nabla \times \vec{H} ) \, dv
        - \int_\Omega \vec{u} \cdot \vec{J} \, dv = \int_\Omega \vec{u} \cdot \vec{S}_j \, dv\\
        & \int_\Omega \vec{u} \cdot \vec{E} \, dv = \int_\Omega \vec{u} \cdot \rho \vec{J} \, dv \\
        & \int_\Omega \vec{u} \cdot \vec{B} \, dv = \int_\Omega \vec{u} \cdot \mu \vec{H} \, dv

    Assuming natural boundary conditions, the surface integral is zero.

    The above expressions are discretized in space according to the finite volume method.
    The discrete magnetic fields :math:`\mathbf{h}` are defined on mesh edges,
    and the discrete current densities :math:`\mathbf{j}` are defined on mesh faces.
    This implies :math:`\mathbf{b}` must be defined on mesh edges and :math:`\mathbf{e}` must
    be defined on mesh faces. Where :math:`\mathbf{u_e}` and :math:`\mathbf{u_f}` represent
    test functions discretized to edges and faces, respectively, we obtain the following
    set of discrete inner-products:

    .. math::
        &\mathbf{u_e^T C^T M_f \, e } + i \omega \mathbf{u_e^T M_e b} = - i\omega \mathbf{u_e^T s_m} \\
        &\mathbf{u_f^T C \, h} - \mathbf{u_f^T j} = \mathbf{u_f^T s_e} \\
        &\mathbf{u_f^T M_f e} = \mathbf{u_f^T M_{f\rho} j} \\
        &\mathbf{u_e^T M_e b} = \mathbf{u_e^T M_{e \mu} h}

    where

    * :math:`\mathbf{C}` is the discrete curl operator
    * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
    * :math:`\mathbf{M_e}` is the edge inner-product matrix
    * :math:`\mathbf{M_f}` is the face inner-product matrix
    * :math:`\mathbf{M_{f\rho}}` is the inner-product matrix for resistivities projected to faces
    * :math:`\mathbf{M_{e\mu}}` is the inner-product matrix for permeabilities projected to edges

    By cancelling like-terms and combining the discrete expressions to solve for the current density, we obtain:

    .. math::
        \mathbf{A \, j} = \mathbf{q}

    where

    * :math:`\mathbf{A} = \mathbf{C M_{e\mu}^{-1} C^T M_{f\rho} + i\omega \mathbf{I}`
    * :math:`\mathbf{q} = - i \omega \mathbf{s_e} - i \omega \mathbf{C M_{e\mu}^{-1} s_m}`

    """

    _solutionType = "jSolution"
    _formulation = "HJ"
    fieldsPair = Fields3DCurrentDensity

    permittivity = props.PhysicalProperty("Dielectric permittivity (F/m)")

    def __init__(
        self, mesh, survey=None, forward_only=False, permittivity=None, **kwargs
    ):
        super().__init__(mesh=mesh, survey=survey, forward_only=forward_only, **kwargs)
        self.permittivity = permittivity

    def getA(self, freq):
        r"""System matrix for the frequency provided.

        This method returns the system matrix for the frequency provided.
        The system matrix at each frequency is given by:

        .. math::
            \mathbf{A} = \mathbf{C M_{e\mu}^{-1} C^T M_{f\rho}} + i\omega \mathbf{I}

        where

        * :math:`\mathbf{M_{f\rho}}` is the inner-product matrix for resistivities projected to faces
        * :math:`\mathbf{M_{e\mu}}` is the inner-product matrix for permeabilities projected to edges

        See the *Notes* section of the doc strings for :class:`Simulation3DCurrentDensity`
        for a full description of the formulation.

        Parameters
        ----------
        freq : float
            The frequency in Hz.

        Returns
        -------
        (n_faces, n_faces) sp.sparse.csr_matrix
            The system matrix.
        """

        MeMuI = self.MeMuI
        MfRho = self.MfRho
        C = self.mesh.edge_curl
        iomega = 1j * omega(freq) * sp.eye(self.mesh.nF)

        if self.permittivity is not None:
            Mfyhati = self._get_face_admittivity_property_matrix(
                freq, invert_model=True
            )
            A = C * MeMuI * C.T.tocsr() * Mfyhati + iomega
        else:
            A = C * MeMuI * C.T.tocsr() * MfRho + iomega

        if self._makeASymmetric is True:
            return MfRho.T.tocsr() * A
        return A

    def getADeriv_rho(self, freq, u, v, adjoint=False):
        r"""Resistivity derivative operation for the system matrix times a vector.

        The system matrix at each frequency is given by:

        .. math::
            \mathbf{A} = \mathbf{C M_{e\mu}^{-1} C^T M_{f\rho}} + i\omega \mathbf{I}

        where

        * :math:`\mathbf{M_{f\rho}}` is the inner-product matrix for resistivities projected to faces
        * :math:`\mathbf{M_{e\mu}}` is the inner-product matrix for permeabilities projected to edges

        See the *Notes* section of the doc strings for :class:`Simulation3DCurrentDensity`
        for a full description of the formulation.

        Where :math:`\mathbf{m}_\boldsymbol{\rho}` are the set of model parameters defining the resistivity,
        :math:`\mathbf{v}` is a vector and :math:`\mathbf{j}` is the discrete current density solution, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A \, j})}{\partial \mathbf{m}_\boldsymbol{\sigma}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A \, j})}{\partial \mathbf{m}_\boldsymbol{\sigma}}^T \, \mathbf{v}

        Parameters
        ----------
        freq : float
            The frequency in Hz.
        u : (n_faces,) numpy.ndarray
            The solution for the fields for the current model at the specified frequency.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_faces,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_faces,) for the standard operation.
            (n_param,) for the adjoint operation.
        """

        MeMuI = self.MeMuI
        C = self.mesh.edge_curl

        if adjoint:
            vec = C * (MeMuI.T * (C.T * v))
            return self.MfRhoDeriv(u, vec, adjoint)
        return C * (MeMuI * (C.T * (self.MfRhoDeriv(u, v, adjoint))))

    def getADeriv_mu(self, freq, u, v, adjoint=False):
        r"""Permeability derivative operation for the system matrix times a vector.

        The system matrix at each frequency is given by:

        .. math::
            \mathbf{A} = \mathbf{C M_{e\mu}^{-1} C^T M_{f\rho}} + i\omega \mathbf{I}

        where

        * :math:`\mathbf{M_{f\rho}}` is the inner-product matrix for resistivities projected to faces
        * :math:`\mathbf{M_{e\mu}}` is the inner-product matrix for permeabilities projected to edges

        See the *Notes* section of the doc strings for :class:`Simulation3DCurrentDensity`
        for a full description of the formulation.

        Where :math:`\mathbf{m}_\boldsymbol{\mu}` are the set of model parameters defining the permeability,
        :math:`\mathbf{v}` is a vector and :math:`\mathbf{j}` is the discrete current density solution, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A \, j})}{\partial \mathbf{m}_\boldsymbol{\mu}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A \, j})}{\partial \mathbf{m}_\boldsymbol{\mu}}^T \, \mathbf{v}

        Parameters
        ----------
        freq : float
            The frequency in Hz.
        u : (n_faces,) numpy.ndarray
            The solution for the fields for the current model at the specified frequency.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_faces,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_faces,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        C = self.mesh.edge_curl
        MfRho = self.MfRho

        MeMuIDeriv = self.MeMuIDeriv(C.T * (MfRho * u))

        if adjoint is True:
            # if self._makeASymmetric:
            #     v = MfRho * v
            return MeMuIDeriv.T * (C.T * v)

        Aderiv = C * (MeMuIDeriv * v)
        # if self._makeASymmetric:
        #     Aderiv = MfRho.T * Aderiv
        return Aderiv

    def getADeriv(self, freq, u, v, adjoint=False):
        r"""Derivative operation for the system matrix times a vector.

        The system matrix at each frequency is given by:

        .. math::
            \mathbf{A} = \mathbf{C M_{e\mu}^{-1} C^T M_{f\rho}} + i\omega \mathbf{I}

        where

        * :math:`\mathbf{M_{f\rho}}` is the inner-product matrix for resistivities projected to faces
        * :math:`\mathbf{M_{e\mu}}` is the inner-product matrix for permeabilities projected to edges

        See the *Notes* section of the doc strings for :class:`Simulation3DCurrentDensity`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters defining the electromagnetic properties,
        :math:`\mathbf{v}` is a vector and :math:`\mathbf{j}` is the discrete current density solution, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A \, j})}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A \, j})}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        freq : float
            The frequency in Hz.
        u : (n_faces,) numpy.ndarray
            The solution for the fields for the current model at the specified frequency.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_faces,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_faces,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        if adjoint and self._makeASymmetric:
            v = self.MfRho * v

        ADeriv = self.getADeriv_rho(freq, u, v, adjoint) + self.getADeriv_mu(
            freq, u, v, adjoint
        )

        if not adjoint and self._makeASymmetric:
            return self.MfRho.T * ADeriv

        return ADeriv

    def getRHS(self, freq):
        r"""Right-hand sides for the given frequency.

        This method returns the right-hand sides for the frequency provided.
        The right-hand side for each source is constructed according to:

        .. math::
            \mathbf{q} = - i \omega \mathbf{s_e} - i \omega \mathbf{C M_{e\mu}^{-1} s_m}

        where

        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
        * :math:`\mathbf{M_{e\mu}}` is the inner-product matrices for permeabilities projected to edges

        See the *Notes* section of the doc strings for :class:`Simulation3DCurrentDensity`
        for a full description of the formulation.

        Parameters
        ----------
        freq : float
            The frequency in Hz.

        Returns
        -------
        (n_faces, n_sources) numpy.ndarray
            The right-hand sides.
        """

        s_m, s_e = self.getSourceTerm(freq)
        C = self.mesh.edge_curl
        MeMuI = self.MeMuI

        RHS = C * (MeMuI * s_m) - 1j * omega(freq) * s_e
        if self._makeASymmetric is True:
            MfRho = self.MfRho
            return MfRho.T * RHS

        return RHS

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        r"""Derivative of the right-hand side times a vector for a given source and frequency.

        The right-hand side for each source is constructed according to:

        .. math::
            \mathbf{q} = - i \omega \mathbf{s_e} - i \omega \mathbf{C M_{e\mu}^{-1} s_m}

        where

        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
        * :math:`\mathbf{M_{e\mu}}` is the inner-product matrices for permeabilities projected to edges

        See the *Notes* section of the doc strings for :class:`Simulation3DCurrentDensity`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters and :math:`\mathbf{v}` is a vector,
        this method returns

        .. math::
            \frac{\partial \mathbf{q_k}}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial \mathbf{q_k}}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        freq : int
            The frequency in Hz.
        src : .frequency_domain.sources.BaseFDEMSrc
            The FDEM source object.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_faces,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of the right-hand sides times a vector. (n_faces,) for the standard operation.
            (n_param,) for the adjoint operation.
        """

        # RHS = C * (MeMuI * s_m) - 1j * omega(freq) * s_e
        # if self._makeASymmetric is True:
        #     MfRho = self.MfRho
        #     return MfRho.T*RHS

        C = self.mesh.edge_curl
        MeMuI = self.MeMuI
        MeMuIDeriv = self.MeMuIDeriv
        s_mDeriv, s_eDeriv = src.evalDeriv(self, adjoint=adjoint)
        s_m, _ = self.getSourceTerm(freq)

        if adjoint:
            if self._makeASymmetric:
                MfRho = self.MfRho
                v = MfRho * v
            CTv = C.T * v
            return (
                s_mDeriv(MeMuI.T * CTv)
                + MeMuIDeriv(s_m).T * CTv
                - 1j * omega(freq) * s_eDeriv(v)
            )

        else:
            RHSDeriv = C * (MeMuI * s_mDeriv(v) + MeMuIDeriv(s_m) * v) - 1j * omega(
                freq
            ) * s_eDeriv(v)

            if self._makeASymmetric:
                MfRho = self.MfRho
                return MfRho.T * RHSDeriv
            return RHSDeriv


class Simulation3DMagneticField(BaseFDEMSimulation):
    r"""3D FDEM simulation in terms of the magnetic field.

    This simulation solves for the magnetic field at each frequency.
    In this formulation, the magnetic fields are defined on mesh edges and the
    current densities are defined on mesh faces; i.e. it is an HJ formulation.
    See the *Notes* section for a comprehensive description of the formulation.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        The mesh.
    survey : .frequency_domain.survey.Survey
        The frequency-domain EM survey.
    forward_only : bool, optional
        If ``True``, the factorization for the inverse of the system matrix at each
        frequency is discarded after the fields are computed at that frequency.
        If ``False``, the factorizations of the system matrices for all frequencies are stored.
    permittivity : (n_cells,) numpy.ndarray, optional
        Dielectric permittivity (F/m) defined on the entire mesh. If ``None``, electric displacement
        is ignored. Please note that `permittivity` is not an invertible property, and that future
        development will result in the deprecation of this propery.
    storeJ : bool, optional
        Whether to compute and store the sensitivity matrix.

    Notes
    -----
    Here, we start with the Maxwell's equations in the frequency-domain where a
    :math:`+i\omega t` Fourier convention is used:

    .. math::
        \begin{align}
        &\nabla \times \vec{E} + i\omega \vec{B} = - i \omega \vec{S}_m \\
        &\nabla \times \vec{H} - \vec{J} = \vec{S}_e
        \end{align}

    where :math:`\vec{S}_e` is an electric source term that defines a source current density,
    and :math:`\vec{S}_m` magnetic source term that defines a source magnetic flux density.
    For now, we neglect displacement current (the `permittivity` attribute is ``None``).
    We define the constitutive relations for the electrical resistivity :math:`\rho`
    and magnetic permeability :math:`\mu` as:

    .. math::
        \vec{E} &= \rho \vec{J} \\
        \vec{B} &= \mu \vec{H}

    We then take the inner products of all previous expressions with a vector test function :math:`\vec{u}`.
    Through vector calculus identities and the divergence theorem, we obtain:

    .. math::
        & \int_\Omega (\nabla \times \vec{u}) \cdot \vec{E} \; dv
        - \oint_{\partial \Omega} \vec{u} \cdot (\vec{E} \times \hat{n} ) \, da
        + i \omega \int_\Omega \vec{u} \cdot \vec{B} \, dv
        = - i \omega \int_\Omega \vec{u} \cdot \vec{S}_m dv \\
        & \int_\Omega \vec{u} \cdot (\nabla \times \vec{H} ) \, dv
        - \int_\Omega \vec{u} \cdot \vec{J} \, dv = \int_\Omega \vec{u} \cdot \vec{S}_j \, dv\\
        & \int_\Omega \vec{u} \cdot \vec{E} \, dv = \int_\Omega \vec{u} \cdot \rho \vec{J} \, dv \\
        & \int_\Omega \vec{u} \cdot \vec{B} \, dv = \int_\Omega \vec{u} \cdot \mu \vec{H} \, dv

    Assuming natural boundary conditions, the surface integral is zero.

    The above expressions are discretized in space according to the finite volume method.
    The discrete magnetic fields :math:`\mathbf{h}` are defined on mesh edges,
    and the discrete current densities :math:`\mathbf{j}` are defined on mesh faces.
    This implies :math:`\mathbf{b}` must be defined on mesh edges and :math:`\mathbf{e}` must
    be defined on mesh faces. Where :math:`\mathbf{u_e}` and :math:`\mathbf{u_f}` represent
    test functions discretized to edges and faces, respectively, we obtain the following
    set of discrete inner-products:

    .. math::
        &\mathbf{u_e^T C^T M_f \, e } + i \omega \mathbf{u_e^T M_e b} = - i\omega \mathbf{u_e^T s_m} \\
        &\mathbf{u_f^T C \, h} - \mathbf{u_f^T j} = \mathbf{u_f^T s_e} \\
        &\mathbf{u_f^T M_f e} = \mathbf{u_f^T M_{f\rho} j} \\
        &\mathbf{u_e^T M_e b} = \mathbf{u_e^T M_{e \mu} h}

    where

    * :math:`\mathbf{C}` is the discrete curl operator
    * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
    * :math:`\mathbf{M_e}` is the edge inner-product matrix
    * :math:`\mathbf{M_f}` is the face inner-product matrix
    * :math:`\mathbf{M_{f\rho}}` is the inner-product matrix for resistivities projected to faces
    * :math:`\mathbf{M_{e\mu}}` is the inner-product matrix for permeabilities projected to edges

    By cancelling like-terms and combining the discrete expressions to solve for the magnetic field, we obtain:

    .. math::
        \mathbf{A \, h} = \mathbf{q}

    where

    * :math:`\mathbf{A} = \mathbf{C^T M_{f\rho} C} + i\omega \mathbf{M_{e\mu}}`
    * :math:`\mathbf{q} = \mathbf{C^T M_{f\rho} s_e} - i\omega \mathbf{s_m}`

    """

    _solutionType = "hSolution"
    _formulation = "HJ"
    fieldsPair = Fields3DMagneticField

    def getA(self, freq):
        r"""System matrix for the frequency provided.

        This method returns the system matrix for the frequency provided.
        The system matrix at each frequency is given by:

        .. math::
            \mathbf{A} = \mathbf{C^T M_{f\rho} C} + i\omega \mathbf{M_{e\mu}}

        where

        * :math:`\mathbf{M_{f\rho}}` is the inner-product matrix for resistivities projected to faces
        * :math:`\mathbf{M_{e\mu}}` is the inner-product matrix for permeabilities projected to edges

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticField`
        for a full description of the formulation.

        Parameters
        ----------
        freq : float
            The frequency in Hz.

        Returns
        -------
        (n_edges, n_edges) sp.sparse.csr_matrix
            The system matrix.
        """

        MeMu = self.MeMu
        C = self.mesh.edge_curl

        if self.permittivity is None:
            MfRho = self.MfRho
            return C.T.tocsr() * (MfRho * C) + 1j * omega(freq) * MeMu
        else:
            Mfyhati = self._get_face_admittivity_property_matrix(
                freq, invert_model=True
            )
            return C.T.tocsr() * (Mfyhati * C) + 1j * omega(freq) * MeMu

    def getADeriv_rho(self, freq, u, v, adjoint=False):
        r"""Resistivity derivative operation for the system matrix times a vector.

        The system matrix at each frequency is given by:

        .. math::
            \mathbf{A} = \mathbf{C^T M_{f\rho} C} + i\omega \mathbf{M_{e\mu}}

        where

        * :math:`\mathbf{M_{f\rho}}` is the inner-product matrix for resistivities projected to faces
        * :math:`\mathbf{M_{e\mu}}` is the inner-product matrix for permeabilities projected to edges

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticField`
        for a full description of the formulation.

        Where :math:`\mathbf{m}_\boldsymbol{\sigma}` are the set of model parameters defining the conductivity,
        :math:`\mathbf{v}` is a vector and :math:`\mathbf{h}` is the discrete magnetic field solution, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A \, h})}{\partial \mathbf{m}_\boldsymbol{\sigma}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A \, h})}{\partial \mathbf{m}_\boldsymbol{\sigma}}^T \, \mathbf{v}

        Parameters
        ----------
        freq : float
            The frequency in Hz.
        u : (n_edges,) numpy.ndarray
            The solution for the fields for the current model at the specified frequency.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_edges,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_edges,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        C = self.mesh.edge_curl
        if adjoint:
            return self.MfRhoDeriv(C * u, C * v, adjoint)
        return C.T * self.MfRhoDeriv(C * u, v, adjoint)

    def getADeriv_mu(self, freq, u, v, adjoint=False):
        r"""Permeability derivative operation for the system matrix times a vector.

        The system matrix at each frequency is given by:

        .. math::
            \mathbf{A} = \mathbf{C^T M_{f\rho} C} + i\omega \mathbf{M_{e\mu}}

        where

        * :math:`\mathbf{M_{f\rho}}` is the inner-product matrix for resistivities projected to faces
        * :math:`\mathbf{M_{e\mu}}` is the inner-product matrix for permeabilities projected to edges

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticField`
        for a full description of the formulation.

        Where :math:`\mathbf{m}_\boldsymbol{\mu}` are the set of model parameters defining the permeability,
        :math:`\mathbf{v}` is a vector and :math:`\mathbf{h}` is the discrete magnetic field solution, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A \, h})}{\partial \mathbf{m}_\boldsymbol{\mu}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A \, h})}{\partial \mathbf{m}_\boldsymbol{\mu}}^T \, \mathbf{v}

        Parameters
        ----------
        freq : float
            The frequency in Hz.
        u : (n_edges,) numpy.ndarray
            The solution for the fields for the current model at the specified frequency.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_edges,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_edges,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        MeMuDeriv = self.MeMuDeriv(u)

        if adjoint is True:
            return 1j * omega(freq) * (MeMuDeriv.T * v)

        return 1j * omega(freq) * (MeMuDeriv * v)

    def getADeriv(self, freq, u, v, adjoint=False):
        r"""Derivative operation for the system matrix times a vector.

        The system matrix at each frequency is given by:

        .. math::
            \mathbf{A} = \mathbf{C^T M_{f\rho} C} + i\omega \mathbf{M_{e\mu}}

        where

        * :math:`\mathbf{M_{f\rho}}` is the inner-product matrix for resistivities projected to faces
        * :math:`\mathbf{M_{e\mu}}` is the inner-product matrix for permeabilities projected to edges

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticField`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters defining the electromagnetic properties,
        :math:`\mathbf{v}` is a vector and :math:`\mathbf{h}` is the discrete electric field solution, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A \, h})}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A \, h})}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        freq : float
            The frequency in Hz.
        u : (n_edges,) numpy.ndarray
            The solution for the fields for the current model at the specified frequency.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_edges,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_edges,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        return self.getADeriv_rho(freq, u, v, adjoint) + self.getADeriv_mu(
            freq, u, v, adjoint
        )

    def getRHS(self, freq):
        r"""Right-hand sides for the given frequency.

        This method returns the right-hand sides for the frequency provided.
        The right-hand side for each source is constructed according to:

        .. math::
            \mathbf{q} = \mathbf{C^T M_{f\rho} s_e} - i\omega \mathbf{s_m}

        where

        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
        * :math:`\mathbf{M_{e\mu}}` is the inner-product matrices for permeabilities projected to edges
        * :math:`\mathbf{M_{f\rho}}` is the inner-product matrices for resistivities projected to faces

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticField`
        for a full description of the formulation.

        Parameters
        ----------
        freq : float
            The frequency in Hz.

        Returns
        -------
        (n_edges, n_sources) numpy.ndarray
            The right-hand sides.
        """

        s_m, s_e = self.getSourceTerm(freq)
        C = self.mesh.edge_curl

        if self.permittivity is None:
            MfRho = self.MfRho
            return s_m + C.T * (MfRho * s_e)
        else:
            Mfyhati = self._get_face_admittivity_property_matrix(
                freq, invert_model=True
            )
            return s_m + C.T * (Mfyhati * s_e)

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        r"""Derivative of the right-hand side times a vector for a given source and frequency.

        The right-hand side for each source is constructed according to:

        .. math::
            \mathbf{q} = \mathbf{C^T M_{f\rho} s_e} - i\omega \mathbf{s_m}

        where

        * :math:`\mathbf{C}` is the discrete curl operator
        * :math:`\mathbf{s_m}` and :math:`\mathbf{s_e}` are the integrated magnetic and electric source terms, respectively
        * :math:`\mathbf{M_{e\mu}}` is the inner-product matrices for permeabilities projected to edges
        * :math:`\mathbf{M_{f\rho}}` is the inner-product matrices for resistivities projected to faces

        See the *Notes* section of the doc strings for :class:`Simulation3DMagneticField`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters and :math:`\mathbf{v}` is a vector,
        this method returns

        .. math::
            \frac{\partial \mathbf{q_k}}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial \mathbf{q_k}}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        freq : int
            The frequency in Hz.
        src : .frequency_domain.sources.BaseFDEMSrc
            The FDEM source object.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_edges,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of the right-hand sides times a vector. (n_edges,) for the standard operation.
            (n_param,) for the adjoint operation.
        """

        _, s_e = src.eval(self)
        C = self.mesh.edge_curl
        MfRho = self.MfRho

        # MfRhoDeriv = self.MfRhoDeriv(s_e)
        # if not adjoint:
        #     RHSDeriv = C.T * (MfRhoDeriv * v)
        # elif adjoint:
        #     RHSDeriv = MfRhoDeriv.T * (C * v)
        if not adjoint:
            RHSDeriv = C.T * (self.MfRhoDeriv(s_e, v, adjoint))
        elif adjoint:
            RHSDeriv = self.MfRhoDeriv(s_e, C * v, adjoint)

        s_mDeriv, s_eDeriv = src.evalDeriv(self, adjoint=adjoint)

        return RHSDeriv + s_mDeriv(v) + C.T * (MfRho * s_eDeriv(v))
