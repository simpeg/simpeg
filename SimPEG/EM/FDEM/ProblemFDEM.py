from SimPEG import Problem, Utils, Props, Solver as SimpegSolver
from .SurveyFDEM import Survey as SurveyFDEM
from .FieldsFDEM import (
    FieldsFDEM, Fields3D_e, Fields3D_b, Fields3D_h, Fields3D_j
)
from SimPEG.EM.Base import BaseEMProblem
from SimPEG.EM.Utils import omega

import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0


class BaseFDEMProblem(BaseEMProblem):
    """
        We start by looking at Maxwell's equations in the electric
        field \\\(\\\mathbf{e}\\\) and the magnetic flux
        density \\\(\\\mathbf{b}\\\)

        .. math ::

            \mathbf{C} \mathbf{e} + i \omega \mathbf{b} = \mathbf{s_m} \\\\
            {\mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{b} -
            \mathbf{M_{\sigma}^e} \mathbf{e} = \mathbf{s_e}}

        if using the E-B formulation (:code:`Problem3D_e`
        or :code:`Problem3D_b`). Note that in this case,
        :math:`\mathbf{s_e}` is an integrated quantity.

        If we write Maxwell's equations in terms of
        \\\(\\\mathbf{h}\\\) and current density \\\(\\\mathbf{j}\\\)

        .. math ::

            \mathbf{C}^{\\top} \mathbf{M_{\\rho}^f} \mathbf{j} +
            i \omega \mathbf{M_{\mu}^e} \mathbf{h} = \mathbf{s_m} \\\\
            \mathbf{C} \mathbf{h} - \mathbf{j} = \mathbf{s_e}

        if using the H-J formulation (:code:`Problem3D_j` or
        :code:`Problem3D_h`). Note that here, :math:`\mathbf{s_m}` is an
        integrated quantity.

        The problem performs the elimination so that we are solving the system
        for \\\(\\\mathbf{e},\\\mathbf{b},\\\mathbf{j} \\\) or
        \\\(\\\mathbf{h}\\\)

    """

    surveyPair = SurveyFDEM
    fieldsPair = FieldsFDEM

    mu, muMap, muDeriv = Props.Invertible(
        "Magnetic Permeability (H/m)",
        default=mu_0
    )

    mui, muiMap, muiDeriv = Props.Invertible(
        "Inverse Magnetic Permeability (m/H)"
    )

    Props.Reciprocal(mu, mui)

    def fields(self, m=None):
        """
        Solve the forward problem for the fields.

        :param numpy.array m: inversion model (nP,)
        :rtype: numpy.array
        :return f: forward solution
        """

        if m is not None:
            self.model = m

        f = self.fieldsPair(self.mesh, self.survey)

        for freq in self.survey.freqs:
            A = self.getA(freq)
            rhs = self.getRHS(freq)
            Ainv = self.Solver(A, **self.solverOpts)
            u = Ainv * rhs
            Srcs = self.survey.getSrcByFreq(freq)
            f[Srcs, self._solutionType] = u
            Ainv.clean()
        return f

    def Jvec(self, m, v, f=None):
        """
        Sensitivity times a vector.

        :param numpy.array m: inversion model (nP,)
        :param numpy.array v: vector which we take sensitivity product with
            (nP,)
        :param SimPEG.EM.FDEM.FieldsFDEM.FieldsFDEM u: fields object
        :rtype: numpy.array
        :return: Jv (ndata,)
        """

        if f is None:
            f = self.fields(m)

        self.model = m

        # Jv = self.dataPair(self.survey)
        Jv = []

        for freq in self.survey.freqs:
            A = self.getA(freq)
            # create the concept of Ainv (actually a solve)
            Ainv = self.Solver(A, **self.solverOpts)

            for src in self.survey.getSrcByFreq(freq):
                u_src = f[src, self._solutionType]
                dA_dm_v = self.getADeriv(freq, u_src, v)
                dRHS_dm_v = self.getRHSDeriv(freq, src, v)
                du_dm_v = Ainv * (- dA_dm_v + dRHS_dm_v)

                for rx in src.rxList:
                    Jv.append(
                        rx.evalDeriv(src, self.mesh, f, du_dm_v=du_dm_v, v=v)
                    )
            Ainv.clean()
        return np.hstack(Jv)

    def Jtvec(self, m, v, f=None):
        """
        Sensitivity transpose times a vector

        :param numpy.array m: inversion model (nP,)
        :param numpy.array v: vector which we take adjoint product with (nP,)
        :param SimPEG.EM.FDEM.FieldsFDEM.FieldsFDEM u: fields object
        :rtype: numpy.array
        :return: Jv (ndata,)
        """

        if f is None:
            f = self.fields(m)

        self.model = m

        # Ensure v is a data object.
        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        Jtv = np.zeros(m.size)

        for freq in self.survey.freqs:
            AT = self.getA(freq).T
            ATinv = self.Solver(AT, **self.solverOpts)

            for src in self.survey.getSrcByFreq(freq):
                u_src = f[src, self._solutionType]

                for rx in src.rxList:
                    df_duT, df_dmT = rx.evalDeriv(
                        src, self.mesh, f, v=v[src, rx], adjoint=True
                    )

                    ATinvdf_duT = ATinv * df_duT

                    dA_dmT = self.getADeriv(
                        freq, u_src, ATinvdf_duT, adjoint=True
                    )
                    dRHS_dmT = self.getRHSDeriv(
                        freq, src, ATinvdf_duT, adjoint=True
                    )
                    du_dmT = -dA_dmT + dRHS_dmT

                    df_dmT = df_dmT + du_dmT

                    # TODO: this should be taken care of by the reciever?
                    if rx.component is 'real':
                        Jtv +=   np.array(df_dmT, dtype=complex).real
                    elif rx.component is 'imag':
                        Jtv += - np.array(df_dmT, dtype=complex).real
                    else:
                        raise Exception('Must be real or imag')

            ATinv.clean()

        return Utils.mkvc(Jtv)

    def getSourceTerm(self, freq):
        """
        Evaluates the sources for a given frequency and puts them in matrix
        form

        :param float freq: Frequency
        :rtype: tuple
        :return: (s_m, s_e) (nE or nF, nSrc)
        """
        Srcs = self.survey.getSrcByFreq(freq)
        if self._formulation is 'EB':
            s_m = np.zeros((self.mesh.nF, len(Srcs)), dtype=complex)
            s_e = np.zeros((self.mesh.nE, len(Srcs)), dtype=complex)
        elif self._formulation is 'HJ':
            s_m = np.zeros((self.mesh.nE, len(Srcs)), dtype=complex)
            s_e = np.zeros((self.mesh.nF, len(Srcs)), dtype=complex)

        for i, src in enumerate(Srcs):
            smi, sei = src.eval(self)

            s_m[:, i] = s_m[:, i] + smi
            s_e[:, i] = s_e[:, i] + sei

        return s_m, s_e


###############################################################################
#                               E-B Formulation                               #
###############################################################################

class Problem3D_e(BaseFDEMProblem):
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

    :param discretize.BaseMesh.BaseMesh mesh: mesh
    """

    _solutionType = 'eSolution'
    _formulation  = 'EB'
    fieldsPair    = Fields3D_e

    def __init__(self, mesh, **kwargs):
        BaseFDEMProblem.__init__(self, mesh, **kwargs)

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
        C = self.mesh.edgeCurl

        return C.T*MfMui*C + 1j*omega(freq)*MeSigma

    # def getADeriv(self, freq, u, v, adjoint=False):
    #     return

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

        dMe_dsig = self.MeSigmaDeriv(u)

        if adjoint:
            return 1j * omega(freq) * ( dMe_dsig.T * v )

        return 1j * omega(freq) * ( dMe_dsig * v )

    def getADeriv_mui(self, freq, u, v, adjoint=False):
        """
        Product of the derivative of the system matrix with respect to the
        permeability model and a vector.

        .. math ::
            \\frac{\mathbf{A}(\mathbf{m}) \mathbf{v}}{d \mathbf{m}_{\\mu^{-1}} =
            \mathbf{C}^{\top} \\frac{d \mathbf{M^f_{\\mu^{-1}}}\mathbf{v}}{d\mathbf{m}}

        """

        C = self.mesh.edgeCurl

        if adjoint:
            return (self.MfMuiDeriv(C*u).T * (C * v))

        return C.T * (self.MfMuiDeriv(C*u) * v)

    def getADeriv(self, freq, u, v, adjoint=False):

        return (
            self.getADeriv_sigma(freq, u, v, adjoint) +
            self.getADeriv_mui(freq, u, v, adjoint)
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
        C = self.mesh.edgeCurl
        MfMui = self.MfMui

        return C.T * (MfMui * s_m) - 1j * omega(freq) * s_e

    def getRHSDeriv(self, freq, src, v, adjoint=False):

        """
        Derivative of the Right-hand side with respect to the model. This
        includes calls to derivatives in the sources
        """

        C = self.mesh.edgeCurl
        MfMui = self.MfMui
        s_m, s_e = self.getSourceTerm(freq)
        s_mDeriv, s_eDeriv = src.evalDeriv(self, adjoint=adjoint)
        MfMuiDeriv = self.MfMuiDeriv(s_m)

        if adjoint:
            return (
                s_mDeriv(MfMui * (C * v)) + MfMuiDeriv.T * (C * v) -
                1j * omega(freq) * s_eDeriv(v)
            )
        return (
            C.T * (MfMui * s_mDeriv(v) + MfMuiDeriv * v) -
            1j * omega(freq) * s_eDeriv(v)
        )


class Problem3D_b(BaseFDEMProblem):
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

    :param discretize.BaseMesh.BaseMesh mesh: mesh
    """

    _solutionType = 'bSolution'
    _formulation = 'EB'
    fieldsPair = Fields3D_b

    def __init__(self, mesh, **kwargs):
        BaseFDEMProblem.__init__(self, mesh, **kwargs)

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
        C = self.mesh.edgeCurl
        iomega = 1j * omega(freq) * sp.eye(self.mesh.nF)

        A = C * (MeSigmaI * (C.T * MfMui)) + iomega

        if self._makeASymmetric is True:
            return MfMui.T*A
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
        C = self.mesh.edgeCurl
        MeSigmaIDeriv = self.MeSigmaIDeriv
        vec = C.T * (MfMui * u)

        MeSigmaIDeriv = MeSigmaIDeriv(vec)

        if adjoint:
            return MeSigmaIDeriv.T * (C.T * v)
        return C * (MeSigmaIDeriv * v)

    def getADeriv_mui(self, freq, u, v, adjoint=False):

        MfMui = self.MfMui
        MfMuiDeriv = self.MfMuiDeriv(u)
        MeSigmaI = self.MeSigmaI
        C = self.mesh.edgeCurl

        if adjoint:
            return MfMuiDeriv.T * (C * (MeSigmaI.T * (C.T * v)))
        return C * (MeSigmaI * (C.T * (MfMuiDeriv * v)))

    def getADeriv(self, freq, u, v, adjoint=False):
        if adjoint is True and self._makeASymmetric:
            v = self.MfMui * v

        ADeriv =  (
            self.getADeriv_sigma(freq, u, v, adjoint) +
            self.getADeriv_mui(freq, u, v, adjoint)
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
        C = self.mesh.edgeCurl
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
        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: FDEM source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of rhs deriv with a vector
        """

        C = self.mesh.edgeCurl
        s_m, s_e = src.eval(self)
        MfMui = self.MfMui

        if self._makeASymmetric and adjoint:
            v = self.MfMui * v

        MeSigmaIDeriv = self.MeSigmaIDeriv(s_e)
        s_mDeriv, s_eDeriv = src.evalDeriv(self, adjoint=adjoint)

        if not adjoint:
            RHSderiv = C * (MeSigmaIDeriv * v)
            SrcDeriv = s_mDeriv(v) + C * (self.MeSigmaI * s_eDeriv(v))
        elif adjoint:
            RHSderiv = MeSigmaIDeriv.T * (C.T * v)
            SrcDeriv = s_mDeriv(v) + s_eDeriv(self.MeSigmaI.T * (C.T * v))

        if self._makeASymmetric is True and not adjoint:
            return MfMui.T * (SrcDeriv + RHSderiv)

        return RHSderiv + SrcDeriv


###############################################################################
#                               H-J Formulation                               #
###############################################################################


class Problem3D_j(BaseFDEMProblem):
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

    :param discretize.BaseMesh.BaseMesh mesh: mesh
    """

    _solutionType = 'jSolution'
    _formulation  = 'HJ'
    fieldsPair    = Fields3D_j

    def __init__(self, mesh, **kwargs):
        BaseFDEMProblem.__init__(self, mesh, **kwargs)

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
        C = self.mesh.edgeCurl
        iomega = 1j * omega(freq) * sp.eye(self.mesh.nF)

        A = C * MeMuI * C.T * MfRho + iomega

        if self._makeASymmetric is True:
            return MfRho.T*A
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
        C = self.mesh.edgeCurl
        MfRhoDeriv = self.MfRhoDeriv(u)

        if adjoint:
            return MfRhoDeriv.T * (C * (MeMuI.T * (C.T * v)))

        return C * (MeMuI * (C.T * (MfRhoDeriv * v)))

    def getADeriv_mu(self, freq, u, v, adjoint=False):

        C = self.mesh.edgeCurl
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

        ADeriv = (
            self.getADeriv_rho(freq, u, v, adjoint) +
            self.getADeriv_mu(freq, u, v, adjoint)
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
        C = self.mesh.edgeCurl
        MeMuI = self.MeMuI

        RHS = C * (MeMuI * s_m) - 1j * omega(freq) * s_e
        if self._makeASymmetric is True:
            MfRho = self.MfRho
            return MfRho.T*RHS

        return RHS

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model

        :param float freq: frequency
        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: FDEM source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of rhs deriv with a vector
        """

        # RHS = C * (MeMuI * s_m) - 1j * omega(freq) * s_e
        # if self._makeASymmetric is True:
        #     MfRho = self.MfRho
        #     return MfRho.T*RHS

        C = self.mesh.edgeCurl
        MeMuI = self.MeMuI
        MeMuIDeriv = self.MeMuIDeriv
        s_mDeriv, s_eDeriv = src.evalDeriv(self, adjoint=adjoint)
        s_m, _ = self.getSourceTerm(freq)

        if adjoint:
            if self._makeASymmetric:
                MfRho = self.MfRho
                v = MfRho*v
            CTv = (C.T * v)
            return (
                s_mDeriv(MeMuI.T * CTv) + MeMuIDeriv(s_m).T * CTv -
                1j * omega(freq) * s_eDeriv(v)
            )

        else:
            RHSDeriv = (
                C * (MeMuI * s_mDeriv(v) + MeMuIDeriv(s_m) * v) -
                1j * omega(freq) * s_eDeriv(v)
            )

            if self._makeASymmetric:
                MfRho = self.MfRho
                return MfRho.T * RHSDeriv
            return RHSDeriv


class Problem3D_h(BaseFDEMProblem):
    """
    We eliminate \\\(\\\mathbf{j}\\\) using

    .. math ::

        \mathbf{j} = \mathbf{C} \mathbf{h} - \mathbf{s_e}

    and solve for \\\(\\\mathbf{h}\\\) using

    .. math ::

        \\left(\mathbf{C}^{\\top} \mathbf{M_{\\rho}^f} \mathbf{C} +
        i \omega \mathbf{M_{\mu}^e}\\right) \mathbf{h} = \mathbf{M^e}
        \mathbf{s_m} + \mathbf{C}^{\\top} \mathbf{M_{\\rho}^f} \mathbf{s_e}

    :param discretize.BaseMesh.BaseMesh mesh: mesh
    """

    _solutionType = 'hSolution'
    _formulation  = 'HJ'
    fieldsPair    = Fields3D_h

    def __init__(self, mesh, **kwargs):
        BaseFDEMProblem.__init__(self, mesh, **kwargs)

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
        C = self.mesh.edgeCurl

        return C.T * (MfRho * C) + 1j*omega(freq)*MeMu

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

        MeMu = self.MeMu
        C = self.mesh.edgeCurl
        MfRhoDeriv = self.MfRhoDeriv(C*u)

        if adjoint:
            return MfRhoDeriv.T * (C * v)
        return C.T * (MfRhoDeriv * v)

    def getADeriv_mu(self, freq, u, v, adjoint=False):
        MeMuDeriv = self.MeMuDeriv(u)

        if adjoint is True:
            return 1j*omega(freq) * (MeMuDeriv.T * v)

        return 1j*omega(freq) * (MeMuDeriv * v)

    def getADeriv(self, freq, u, v, adjoint=False):
        return (
            self.getADeriv_rho(freq, u, v, adjoint) +
            self.getADeriv_mu(freq, u, v, adjoint)
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
        C = self.mesh.edgeCurl
        MfRho = self.MfRho

        return s_m + C.T * (MfRho * s_e)

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model

        :param float freq: frequency
        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: FDEM source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of rhs deriv with a vector
        """

        _, s_e = src.eval(self)
        C = self.mesh.edgeCurl
        MfRho = self.MfRho

        MfRhoDeriv = self.MfRhoDeriv(s_e)
        if not adjoint:
            RHSDeriv = C.T * (MfRhoDeriv * v)
        elif adjoint:
            RHSDeriv = MfRhoDeriv.T * (C * v)

        s_mDeriv, s_eDeriv = src.evalDeriv(self, adjoint=adjoint)

        return RHSDeriv + s_mDeriv(v) + C.T * (MfRho * s_eDeriv(v))
