import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0
import properties
from discretize.utils import Zero

from ... import props
from ...data import Data
from ...utils import mkvc
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


class BaseFDEMSimulation(BaseEMSimulation):
    """
    We start by looking at Maxwell's equations in the electric
    field \\\(\\\mathbf{e}\\\) and the magnetic flux
    density \\\(\\\mathbf{b}\\\)

    .. math ::

        \mathbf{C} \mathbf{e} + i \omega \mathbf{b} = \mathbf{s_m} \\\\
        {\mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{b} -
        \mathbf{M_{\sigma}^e} \mathbf{e} = \mathbf{s_e}}

    if using the E-B formulation (:code:`Simulation3DElectricField`
    or :code:`Simulation3DMagneticFluxDensity`). Note that in this case,
    :math:`\mathbf{s_e}` is an integrated quantity.

    If we write Maxwell's equations in terms of
    \\\(\\\mathbf{h}\\\) and current density \\\(\\\mathbf{j}\\\)

    .. math ::

        \mathbf{C}^{\\top} \mathbf{M_{\\rho}^f} \mathbf{j} +
        i \omega \mathbf{M_{\mu}^e} \mathbf{h} = \mathbf{s_m} \\\\
        \mathbf{C} \mathbf{h} - \mathbf{j} = \mathbf{s_e}

    if using the H-J formulation (:code:`Simulation3DCurrentDensity` or
    :code:`Simulation3DMagneticField`). Note that here, :math:`\mathbf{s_m}` is an
    integrated quantity.

    The problem performs the elimination so that we are solving the system
    for \\\(\\\mathbf{e},\\\mathbf{b},\\\mathbf{j} \\\) or
    \\\(\\\mathbf{h}\\\)

    """

    fieldsPair = FieldsFDEM

    mu, muMap, muDeriv = props.Invertible("Magnetic Permeability (H/m)", default=mu_0)

    mui, muiMap, muiDeriv = props.Invertible("Inverse Magnetic Permeability (m/H)")

    props.Reciprocal(mu, mui)

    forward_only = properties.Boolean(
        "If True, A-inverse not stored at each frequency in forward simulation",
        default=False,
    )

    survey = properties.Instance("a survey object", Survey, required=True)

    # @profile
    def fields(self, m=None):
        """
        Solve the forward problem for the fields.

        :param numpy.ndarray m: inversion model (nP,)
        :rtype: numpy.ndarray
        :return f: forward solution
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
        """
        Sensitivity times a vector.

        :param numpy.ndarray m: inversion model (nP,)
        :param numpy.ndarray v: vector which we take sensitivity product with
            (nP,)
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM u: fields object
        :rtype: numpy.ndarray
        :return: Jv (ndata,)
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
        """
        Sensitivity transpose times a vector

        :param numpy.ndarray m: inversion model (nP,)
        :param numpy.ndarray v: vector which we take adjoint product with (nP,)
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM u: fields object
        :rtype: numpy.ndarray
        :return: Jv (ndata,)
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

    # @profile
    def getSourceTerm(self, freq):
        """
        Evaluates the sources for a given frequency and puts them in matrix
        form

        :param float freq: Frequency
        :rtype: tuple
        :return: (s_m, s_e) (nE or nF, nSrc)
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


###############################################################################
#                               E-B Formulation                               #
###############################################################################


class Simulation3DElectricField(BaseFDEMSimulation):
    """
    By eliminating the magnetic flux density using

        .. math ::

            \mathbf{b} = \\frac{1}{i \omega}\\left(-\mathbf{C} \mathbf{e} +
            \mathbf{s_m}\\right)


    we can write Maxwell's equations as a second order system in
    \\\(\\\mathbf{e}\\\) only:

    .. math ::

        \\left(\mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{C} +
        i \omega \mathbf{M^e_{\sigma}} \\right)\mathbf{e} =
        \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f}\mathbf{s_m}
        - i\omega\mathbf{M^e}\mathbf{s_e}

    which we solve for :math:`\mathbf{e}`.

    :param discretize.base.BaseMesh mesh: mesh
    """

    _solutionType = "eSolution"
    _formulation = "EB"
    fieldsPair = Fields3DElectricField

    def __init__(self, mesh, **kwargs):
        super(Simulation3DElectricField, self).__init__(mesh, **kwargs)

    def getA(self, freq):
        """
        System matrix

        .. math ::
            \mathbf{A} = \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{C}
            + i \omega \mathbf{M^e_{\sigma}}

        :param float freq: Frequency
        :rtype: scipy.sparse.csr_matrix
        :return: A
        """

        MfMui = self.MfMui
        MeSigma = self.MeSigma
        C = self.mesh.edge_curl

        return C.T.tocsr() * MfMui * C + 1j * omega(freq) * MeSigma

    def getADeriv_sigma(self, freq, u, v, adjoint=False):
        """
        Product of the derivative of our system matrix with respect to the
        conductivity model and a vector

        .. math ::
            \\frac{\mathbf{A}(\mathbf{m}) \mathbf{v}}{d \mathbf{m}_{\\sigma}} =
            i \omega \\frac{d \mathbf{M^e_{\sigma}}(\mathbf{u})\mathbf{v} }{d\mathbf{m}}

        :param float freq: frequency
        :param numpy.ndarray u: solution vector (nE,)
        :param numpy.ndarray v: vector to take prodct with (nP,) or (nD,) for
            adjoint
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative of the system matrix times a vector (nP,) or
            adjoint (nD,)
        """

        dMe_dsig_v = self.MeSigmaDeriv(u, v, adjoint)
        return 1j * omega(freq) * dMe_dsig_v

    def getADeriv_mui(self, freq, u, v, adjoint=False):
        """
        Product of the derivative of the system matrix with respect to the
        permeability model and a vector.

        .. math ::
            \\frac{\mathbf{A}(\mathbf{m}) \mathbf{v}}{d \mathbf{m}_{\\mu^{-1}} =
            \mathbf{C}^{\top} \\frac{d \mathbf{M^f_{\\mu^{-1}}}\mathbf{v}}{d\mathbf{m}}

        """

        C = self.mesh.edge_curl

        if adjoint:
            return self.MfMuiDeriv(C * u).T * (C * v)

        return C.T * (self.MfMuiDeriv(C * u) * v)

    def getADeriv(self, freq, u, v, adjoint=False):

        return self.getADeriv_sigma(freq, u, v, adjoint) + self.getADeriv_mui(
            freq, u, v, adjoint
        )

    def getRHS(self, freq):
        """
        Right hand side for the system

        .. math ::
            \mathbf{RHS} = \mathbf{C}^{\\top}
            \mathbf{M_{\mu^{-1}}^f}\mathbf{s_m} -
            i\omega\mathbf{M_e}\mathbf{s_e}

        :param float freq: Frequency
        :rtype: numpy.ndarray
        :return: RHS (nE, nSrc)
        """

        s_m, s_e = self.getSourceTerm(freq)
        C = self.mesh.edge_curl
        MfMui = self.MfMui

        return C.T * (MfMui * s_m) - 1j * omega(freq) * s_e

    def getRHSDeriv(self, freq, src, v, adjoint=False):

        """
        Derivative of the Right-hand side with respect to the model. This
        includes calls to derivatives in the sources
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
    """
    We eliminate :math:`\mathbf{e}` using

    .. math ::

         \mathbf{e} = \mathbf{M^e_{\sigma}}^{-1} \\left(\mathbf{C}^{\\top}
         \mathbf{M_{\mu^{-1}}^f} \mathbf{b} - \mathbf{s_e}\\right)

    and solve for :math:`\mathbf{b}` using:

    .. math ::

        \\left(\mathbf{C} \mathbf{M^e_{\sigma}}^{-1} \mathbf{C}^{\\top}
        \mathbf{M_{\mu^{-1}}^f}  + i \omega \\right)\mathbf{b} = \mathbf{s_m} +
        \mathbf{M^e_{\sigma}}^{-1}\mathbf{M^e}\mathbf{s_e}

    .. note ::
        The inverse problem will not work with full anisotropy

    :param discretize.base.BaseMesh mesh: mesh
    """

    _solutionType = "bSolution"
    _formulation = "EB"
    fieldsPair = Fields3DMagneticFluxDensity

    def __init__(self, mesh, **kwargs):
        super(Simulation3DMagneticFluxDensity, self).__init__(mesh, **kwargs)

    def getA(self, freq):
        """
        System matrix

        .. math ::
            \mathbf{A} = \mathbf{C} \mathbf{M^e_{\sigma}}^{-1}
            \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f}  + i \omega

        :param float freq: Frequency
        :rtype: scipy.sparse.csr_matrix
        :return: A
        """

        MfMui = self.MfMui
        MeSigmaI = self.MeSigmaI
        C = self.mesh.edge_curl
        iomega = 1j * omega(freq) * sp.eye(self.mesh.nF)

        A = C * (MeSigmaI * (C.T.tocsr() * MfMui)) + iomega

        if self._makeASymmetric:
            return MfMui.T.tocsr() * A
        return A

    def getADeriv_sigma(self, freq, u, v, adjoint=False):

        """
        Product of the derivative of our system matrix with respect to the
        model and a vector

        .. math ::
            \\frac{\mathbf{A}(\mathbf{m}) \mathbf{v}}{d \mathbf{m}} =
            \mathbf{C} \\frac{\mathbf{M^e_{\sigma}} \mathbf{v}}{d\mathbf{m}}

        :param float freq: frequency
        :param numpy.ndarray u: solution vector (nF,)
        :param numpy.ndarray v: vector to take prodct with (nP,) or (nD,) for
            adjoint
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative of the system matrix times a vector (nP,) or
            adjoint (nD,)
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

        MfMui = self.MfMui
        MfMuiDeriv = self.MfMuiDeriv(u)
        MeSigmaI = self.MeSigmaI
        C = self.mesh.edge_curl

        if adjoint:
            return MfMuiDeriv.T * (C * (MeSigmaI.T * (C.T * v)))
        return C * (MeSigmaI * (C.T * (MfMuiDeriv * v)))

    def getADeriv(self, freq, u, v, adjoint=False):
        if adjoint is True and self._makeASymmetric:
            v = self.MfMui * v

        ADeriv = self.getADeriv_sigma(freq, u, v, adjoint) + self.getADeriv_mui(
            freq, u, v, adjoint
        )

        if adjoint is False and self._makeASymmetric:
            return self.MfMui.T * ADeriv

        return ADeriv

    def getRHS(self, freq):
        """
        Right hand side for the system

        .. math ::
            \mathbf{RHS} = \mathbf{s_m} +
            \mathbf{M^e_{\sigma}}^{-1}\mathbf{s_e}

        :param float freq: Frequency
        :rtype: numpy.ndarray
        :return: RHS (nE, nSrc)
        """

        s_m, s_e = self.getSourceTerm(freq)
        C = self.mesh.edge_curl
        MeSigmaI = self.MeSigmaI

        RHS = s_m + C * (MeSigmaI * s_e)

        if self._makeASymmetric is True:
            MfMui = self.MfMui
            return MfMui.T * RHS

        return RHS

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model

        :param float freq: frequency
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM src: FDEM source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of rhs deriv with a vector
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
    """
    We eliminate \\\(\\\mathbf{h}\\\) using

    .. math ::

        \mathbf{h} = \\frac{1}{i \omega} \mathbf{M_{\mu}^e}^{-1}
        \\left(-\mathbf{C}^{\\top} \mathbf{M_{\\rho}^f} \mathbf{j} +
        \mathbf{M^e} \mathbf{s_m} \\right)


    and solve for \\\(\\\mathbf{j}\\\) using

    .. math ::

        \\left(\mathbf{C} \mathbf{M_{\mu}^e}^{-1} \mathbf{C}^{\\top}
        \mathbf{M_{\\rho}^f} + i \omega\\right)\mathbf{j} =
        \mathbf{C} \mathbf{M_{\mu}^e}^{-1} \mathbf{M^e} \mathbf{s_m} -
        i\omega\mathbf{s_e}

    .. note::
        This implementation does not yet work with full anisotropy!!

    :param discretize.base.BaseMesh mesh: mesh
    """

    _solutionType = "jSolution"
    _formulation = "HJ"
    fieldsPair = Fields3DCurrentDensity

    def __init__(self, mesh, **kwargs):
        super(Simulation3DCurrentDensity, self).__init__(mesh, **kwargs)

    def getA(self, freq):
        """
        System matrix

        .. math ::
                \\mathbf{A} = \\mathbf{C}  \\mathbf{M^e_{\\mu^{-1}}}
                \\mathbf{C}^{\\top} \\mathbf{M^f_{\\sigma^{-1}}}  + i\\omega

        :param float freq: Frequency
        :rtype: scipy.sparse.csr_matrix
        :return: A
        """

        MeMuI = self.MeMuI
        MfRho = self.MfRho
        C = self.mesh.edge_curl
        iomega = 1j * omega(freq) * sp.eye(self.mesh.nF)

        A = C * MeMuI * C.T.tocsr() * MfRho + iomega

        if self._makeASymmetric is True:
            return MfRho.T.tocsr() * A
        return A

    def getADeriv_rho(self, freq, u, v, adjoint=False):
        """
        Product of the derivative of our system matrix with respect to the
        model and a vector

        In this case, we assume that electrical conductivity, :math:`\sigma`
        is the physical property of interest (i.e. :math:`\sigma` =
        model.transform). Then we want

        .. math ::

            \\frac{\mathbf{A(\sigma)} \mathbf{v}}{d \mathbf{m}} =
            \mathbf{C} \mathbf{M^e_{mu^{-1}}} \mathbf{C^{\\top}}
            \\frac{d \mathbf{M^f_{\sigma^{-1}}}\mathbf{v} }{d \mathbf{m}}

        :param float freq: frequency
        :param numpy.ndarray u: solution vector (nF,)
        :param numpy.ndarray v: vector to take prodct with (nP,) or (nD,) for
            adjoint
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative of the system matrix times a vector (nP,) or
            adjoint (nD,)
        """

        MeMuI = self.MeMuI
        MfRho = self.MfRho
        C = self.mesh.edge_curl

        if adjoint:
            vec = C * (MeMuI.T * (C.T * v))
            return self.MfRhoDeriv(u, vec, adjoint)
        return C * (MeMuI * (C.T * (self.MfRhoDeriv(u, v, adjoint))))

        # MfRhoDeriv = self.MfRhoDeriv(u)
        # if adjoint:
        #     return MfRhoDeriv.T * (C * (MeMuI.T * (C.T * v)))

        # return C * (MeMuI * (C.T * (MfRhoDeriv * v)))

    def getADeriv_mu(self, freq, u, v, adjoint=False):

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
        if adjoint and self._makeASymmetric:
            v = self.MfRho * v

        ADeriv = self.getADeriv_rho(freq, u, v, adjoint) + self.getADeriv_mu(
            freq, u, v, adjoint
        )

        if not adjoint and self._makeASymmetric:
            return self.MfRho.T * ADeriv

        return ADeriv

    def getRHS(self, freq):
        """
        Right hand side for the system

        .. math ::

            \mathbf{RHS} = \mathbf{C} \mathbf{M_{\mu}^e}^{-1}\mathbf{s_m}
            - i\omega \mathbf{s_e}

        :param float freq: Frequency
        :rtype: numpy.ndarray
        :return: RHS (nE, nSrc)
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
        """
        Derivative of the right hand side with respect to the model

        :param float freq: frequency
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM src: FDEM source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of rhs deriv with a vector
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
    """
    We eliminate \\\(\\\mathbf{j}\\\) using

    .. math ::

        \mathbf{j} = \mathbf{C} \mathbf{h} - \mathbf{s_e}

    and solve for \\\(\\\mathbf{h}\\\) using

    .. math ::

        \\left(\mathbf{C}^{\\top} \mathbf{M_{\\rho}^f} \mathbf{C} +
        i \omega \mathbf{M_{\mu}^e}\\right) \mathbf{h} = \mathbf{M^e}
        \mathbf{s_m} + \mathbf{C}^{\\top} \mathbf{M_{\\rho}^f} \mathbf{s_e}

    :param discretize.base.BaseMesh mesh: mesh
    """

    _solutionType = "hSolution"
    _formulation = "HJ"
    fieldsPair = Fields3DMagneticField

    def __init__(self, mesh, **kwargs):
        super(Simulation3DMagneticField, self).__init__(mesh, **kwargs)

    def getA(self, freq):
        """
        System matrix

        .. math::
            \mathbf{A} = \mathbf{C}^{\\top} \mathbf{M_{\\rho}^f} \mathbf{C} +
            i \omega \mathbf{M_{\mu}^e}


        :param float freq: Frequency
        :rtype: scipy.sparse.csr_matrix
        :return: A

        """

        MeMu = self.MeMu
        MfRho = self.MfRho
        C = self.mesh.edge_curl

        return C.T.tocsr() * (MfRho * C) + 1j * omega(freq) * MeMu

    def getADeriv_rho(self, freq, u, v, adjoint=False):
        """
        Product of the derivative of our system matrix with respect to the
        model and a vector

        .. math::
            \\frac{\mathbf{A}(\mathbf{m}) \mathbf{v}}{d \mathbf{m}} =
            \mathbf{C}^{\\top}\\frac{d \mathbf{M^f_{\\rho}}\mathbf{v}}
            {d\mathbf{m}}

        :param float freq: frequency
        :param numpy.ndarray u: solution vector (nE,)
        :param numpy.ndarray v: vector to take prodct with (nP,) or (nD,) for
            adjoint
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative of the system matrix times a vector (nP,) or
            adjoint (nD,)
        """
        C = self.mesh.edge_curl
        if adjoint:
            return self.MfRhoDeriv(C * u, C * v, adjoint)
        return C.T * self.MfRhoDeriv(C * u, v, adjoint)

    def getADeriv_mu(self, freq, u, v, adjoint=False):
        MeMuDeriv = self.MeMuDeriv(u)

        if adjoint is True:
            return 1j * omega(freq) * (MeMuDeriv.T * v)

        return 1j * omega(freq) * (MeMuDeriv * v)

    def getADeriv(self, freq, u, v, adjoint=False):
        return self.getADeriv_rho(freq, u, v, adjoint) + self.getADeriv_mu(
            freq, u, v, adjoint
        )

    def getRHS(self, freq):
        """
        Right hand side for the system

        .. math ::

            \mathbf{RHS} = \mathbf{M^e} \mathbf{s_m} + \mathbf{C}^{\\top}
            \mathbf{M_{\\rho}^f} \mathbf{s_e}

        :param float freq: Frequency
        :rtype: numpy.ndarray
        :return: RHS (nE, nSrc)

        """

        s_m, s_e = self.getSourceTerm(freq)
        C = self.mesh.edge_curl
        MfRho = self.MfRho

        return s_m + C.T * (MfRho * s_e)

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model

        :param float freq: frequency
        :param SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM src: FDEM source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of rhs deriv with a vector
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
