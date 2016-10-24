from SimPEG import Problem, Utils, np, sp, Solver as SimpegSolver
from scipy.constants import mu_0
from .SurveyFDEM import Survey as SurveyFDEM
from .FieldsFDEM import FieldsFDEM, Fields3D_e, Fields3D_b, Fields3D_h, Fields3D_j
from SimPEG.EM.Base import BaseEMProblem
from SimPEG.EM.Utils import omega


class BaseFDEMProblem(BaseEMProblem):
    """
        We start by looking at Maxwell's equations in the electric
        field \\\(\\\mathbf{e}\\\) and the magnetic flux
        density \\\(\\\mathbf{b}\\\)

        .. math ::

            \mathbf{C} \mathbf{e} + i \omega \mathbf{b} = \mathbf{s_m} \\\\
            {\mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{b} - \mathbf{M_{\sigma}^e} \mathbf{e} = \mathbf{s_e}}

        if using the E-B formulation (:code:`Problem3D_e`
        or :code:`Problem3D_b`). Note that in this case, :math:`\mathbf{s_e}` is an integrated quantity.

        If we write Maxwell's equations in terms of
        \\\(\\\mathbf{h}\\\) and current density \\\(\\\mathbf{j}\\\)

        .. math ::

            \mathbf{C}^{\\top} \mathbf{M_{\\rho}^f} \mathbf{j} + i \omega \mathbf{M_{\mu}^e} \mathbf{h} = \mathbf{s_m} \\\\
            \mathbf{C} \mathbf{h} - \mathbf{j} = \mathbf{s_e}

        if using the H-J formulation (:code:`Problem3D_j` or :code:`Problem3D_h`). Note that here, :math:`\mathbf{s_m}` is an integrated quantity.

        The problem performs the elimination so that we are solving the system for \\\(\\\mathbf{e},\\\mathbf{b},\\\mathbf{j} \\\) or \\\(\\\mathbf{h}\\\)

    """

    surveyPair = SurveyFDEM
    fieldsPair = FieldsFDEM

    def fields(self, m):
        """
        Solve the forward problem for the fields.

        :param numpy.array m: inversion model (nP,)
        :rtype: numpy.array
        :return f: forward solution
        """

        self.curModel = m
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
        :param numpy.array v: vector which we take sensitivity product with (nP,)
        :param SimPEG.EM.FDEM.FieldsFDEM.FieldsFDEM u: fields object
        :rtype: numpy.array
        :return: Jv (ndata,)
        """

        if f is None:
           f = self.fields(m)

        self.curModel = m

        # Jv = self.dataPair(self.survey)
        Jv = []

        for freq in self.survey.freqs:
            A = self.getA(freq)
            Ainv = self.Solver(A, **self.solverOpts) # create the concept of Ainv (actually a solve)

            for src in self.survey.getSrcByFreq(freq):
                u_src = f[src, self._solutionType]
                dA_dm_v = self.getADeriv(freq, u_src, v)
                dRHS_dm_v = self.getRHSDeriv(freq, src, v)
                du_dm_v = Ainv * ( - dA_dm_v + dRHS_dm_v )

                for rx in src.rxList:
                    df_dmFun = getattr(f, '_{0}Deriv'.format(rx.projField), None)
                    df_dm_v = df_dmFun(src, du_dm_v, v, adjoint=False)
                    # Jv[src, rx] = rx.evalDeriv(src, self.mesh, f, df_dm_v)
                    Jv.append(rx.evalDeriv(src, self.mesh, f, df_dm_v))
            Ainv.clean()
        # return Utils.mkvc(Jv)
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

        self.curModel = m

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
                    PTv = rx.evalDeriv(src, self.mesh, f, v[src, rx], adjoint=True) # wrt f, need possibility wrt m

                    df_duTFun = getattr(f, '_{0}Deriv'.format(rx.projField), None)
                    df_duT, df_dmT = df_duTFun(src, None, PTv, adjoint=True)

                    ATinvdf_duT = ATinv * df_duT

                    dA_dmT = self.getADeriv(freq, u_src, ATinvdf_duT, adjoint=True)
                    dRHS_dmT = self.getRHSDeriv(freq, src, ATinvdf_duT, adjoint=True)
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
        Evaluates the sources for a given frequency and puts them in matrix form

        :param float freq: Frequency
        :rtype: tuple
        :return: (s_m, s_e) (nE or nF, nSrc)
        """
        Srcs = self.survey.getSrcByFreq(freq)
        if self._formulation is 'EB':
            s_m = np.zeros((self.mesh.nF,len(Srcs)), dtype=complex)
            s_e = np.zeros((self.mesh.nE,len(Srcs)), dtype=complex)
        elif self._formulation is 'HJ':
            s_m = np.zeros((self.mesh.nE,len(Srcs)), dtype=complex)
            s_e = np.zeros((self.mesh.nF,len(Srcs)), dtype=complex)

        for i, src in enumerate(Srcs):
            smi, sei = src.eval(self)
            #Why are you adding?
            s_m[:,i] = s_m[:,i] + smi
            s_e[:,i] = s_e[:,i] + sei

        return s_m, s_e


##########################################################################################
################################ E-B Formulation #########################################
##########################################################################################

class Problem3D_e(BaseFDEMProblem):
    """
    By eliminating the magnetic flux density using

        .. math ::

            \mathbf{b} = \\frac{1}{i \omega}\\left(-\mathbf{C} \mathbf{e} + \mathbf{s_m}\\right)


    we can write Maxwell's equations as a second order system in \\\(\\\mathbf{e}\\\) only:

    .. math ::

        \\left(\mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{C}+ i \omega \mathbf{M^e_{\sigma}} \\right)\mathbf{e} = \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f}\mathbf{s_m} -i\omega\mathbf{M^e}\mathbf{s_e}

    which we solve for :math:`\mathbf{e}`.

    :param SimPEG.Mesh.BaseMesh.BaseMesh mesh: mesh
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
            \mathbf{A} = \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{C} + i \omega \mathbf{M^e_{\sigma}}

        :param float freq: Frequency
        :rtype: scipy.sparse.csr_matrix
        :return: A
        """

        MfMui = self.MfMui
        MeSigma = self.MeSigma
        C = self.mesh.edgeCurl

        return C.T*MfMui*C + 1j*omega(freq)*MeSigma


    def getADeriv(self, freq, u, v, adjoint=False):
        """
        Product of the derivative of our system matrix with respect to the model and a vector

        .. math ::
            \\frac{\mathbf{A}(\mathbf{m}) \mathbf{v}}{d \mathbf{m}} = i \omega \\frac{d \mathbf{M^e_{\sigma}}\mathbf{v} }{d\mathbf{m}}

        :param float freq: frequency
        :param numpy.ndarray u: solution vector (nE,)
        :param numpy.ndarray v: vector to take prodct with (nP,) or (nD,) for adjoint
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative of the system matrix times a vector (nP,) or adjoint (nD,)
        """

        dsig_dm = self.curModel.sigmaDeriv
        dMe_dsig = self.MeSigmaDeriv(u)

        if adjoint:
            return 1j * omega(freq) * ( dMe_dsig.T * v )

        return 1j * omega(freq) * ( dMe_dsig * v )

    def getRHS(self, freq):
        """
        Right hand side for the system

        .. math ::
            \mathbf{RHS} = \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f}\mathbf{s_m} -i\omega\mathbf{M_e}\mathbf{s_e}

        :param float freq: Frequency
        :rtype: numpy.ndarray
        :return: RHS (nE, nSrc)
        """

        s_m, s_e = self.getSourceTerm(freq)
        C = self.mesh.edgeCurl
        MfMui = self.MfMui

        return C.T * (MfMui * s_m) -1j * omega(freq) * s_e

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model

        :param float freq: frequency
        :param SimPEG.EM.FDEM.SrcFDEM.BaseSrc src: FDEM source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of rhs deriv with a vector
        """

        C = self.mesh.edgeCurl
        MfMui = self.MfMui
        s_mDeriv, s_eDeriv = src.evalDeriv(self, adjoint=adjoint)

        if adjoint:
            dRHS = MfMui * (C * v)
            return s_mDeriv(dRHS) - 1j * omega(freq) * s_eDeriv(v)

        else:
            return C.T * (MfMui * s_mDeriv(v)) -1j * omega(freq) * s_eDeriv(v)


class Problem3D_b(BaseFDEMProblem):
    """
    We eliminate :math:`\mathbf{e}` using

    .. math ::

         \mathbf{e} = \mathbf{M^e_{\sigma}}^{-1} \\left(\mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{b} - \mathbf{s_e}\\right)

    and solve for :math:`\mathbf{b}` using:

    .. math ::

        \\left(\mathbf{C} \mathbf{M^e_{\sigma}}^{-1} \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f}  + i \omega \\right)\mathbf{b} = \mathbf{s_m} + \mathbf{M^e_{\sigma}}^{-1}\mathbf{M^e}\mathbf{s_e}

    .. note ::
        The inverse problem will not work with full anisotropy

    :param SimPEG.Mesh.BaseMesh.BaseMesh mesh: mesh
    """

    _solutionType = 'bSolution'
    _formulation  = 'EB'
    fieldsPair    = Fields3D_b

    def __init__(self, mesh, **kwargs):
        BaseFDEMProblem.__init__(self, mesh, **kwargs)

    def getA(self, freq):
        """
        System matrix

        .. math ::
            \mathbf{A} = \mathbf{C} \mathbf{M^e_{\sigma}}^{-1} \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f}  + i \omega

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

    def getADeriv(self, freq, u, v, adjoint=False):

        """
        Product of the derivative of our system matrix with respect to the model and a vector

        .. math ::
            \\frac{\mathbf{A}(\mathbf{m}) \mathbf{v}}{d \mathbf{m}} = \mathbf{C} \\frac{\mathbf{M^e_{\sigma}} \mathbf{v}}{d\mathbf{m}}

        :param float freq: frequency
        :param numpy.ndarray u: solution vector (nF,)
        :param numpy.ndarray v: vector to take prodct with (nP,) or (nD,) for adjoint
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative of the system matrix times a vector (nP,) or adjoint (nD,)
        """

        MfMui = self.MfMui
        C = self.mesh.edgeCurl
        MeSigmaIDeriv = self.MeSigmaIDeriv
        vec = C.T * (MfMui * u)

        MeSigmaIDeriv = MeSigmaIDeriv(vec)

        if adjoint:
            if self._makeASymmetric is True:
                v = MfMui * v
            return MeSigmaIDeriv.T * (C.T * v)

        if self._makeASymmetric is True:
            return MfMui.T * ( C * ( MeSigmaIDeriv * v ) )
        return C * ( MeSigmaIDeriv * v )


    def getRHS(self, freq):
        """
        Right hand side for the system

        .. math ::
            \mathbf{RHS} = \mathbf{s_m} + \mathbf{M^e_{\sigma}}^{-1}\mathbf{s_e}

        :param float freq: Frequency
        :rtype: numpy.ndarray
        :return: RHS (nE, nSrc)
        """

        s_m, s_e = self.getSourceTerm(freq)
        C = self.mesh.edgeCurl
        MeSigmaI = self.MeSigmaI

        RHS = s_m + C * ( MeSigmaI * s_e )

        if self._makeASymmetric is True:
            MfMui = self.MfMui
            return MfMui.T * RHS

        return RHS

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model

        :param float freq: frequency
        :param SimPEG.EM.FDEM.SrcFDEM.BaseSrc src: FDEM source
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
            SrcDeriv = s_mDeriv(v) + self.MeSigmaI.T * (C.T * s_eDeriv(v))

        if self._makeASymmetric is True and not adjoint:
            return MfMui.T * (SrcDeriv + RHSderiv)

        return RHSderiv + SrcDeriv



##########################################################################################
################################ H-J Formulation #########################################
##########################################################################################


class Problem3D_j(BaseFDEMProblem):
    """
    We eliminate \\\(\\\mathbf{h}\\\) using

    .. math ::

        \mathbf{h} = \\frac{1}{i \omega} \mathbf{M_{\mu}^e}^{-1} \\left(-\mathbf{C}^{\\top} \mathbf{M_{\\rho}^f} \mathbf{j}  + \mathbf{M^e} \mathbf{s_m} \\right)


    and solve for \\\(\\\mathbf{j}\\\) using

    .. math ::

        \\left(\mathbf{C} \mathbf{M_{\mu}^e}^{-1} \mathbf{C}^{\\top} \mathbf{M_{\\rho}^f} + i \omega\\right)\mathbf{j} = \mathbf{C} \mathbf{M_{\mu}^e}^{-1} \mathbf{M^e} \mathbf{s_m} -i\omega\mathbf{s_e}

    .. note::
        This implementation does not yet work with full anisotropy!!

    :param SimPEG.Mesh.BaseMesh.BaseMesh mesh: mesh
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
                \\mathbf{A} = \\mathbf{C}  \\mathbf{M^e_{\\mu^{-1}}} \\mathbf{C}^{\\top} \\mathbf{M^f_{\\sigma^{-1}}}  + i\\omega

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


    def getADeriv(self, freq, u, v, adjoint=False):
        """
        Product of the derivative of our system matrix with respect to the model and a vector

        In this case, we assume that electrical conductivity, :math:`\sigma` is the physical property of interest (i.e. :math:`\sigma` = model.transform). Then we want

        .. math ::

            \\frac{\mathbf{A(\sigma)} \mathbf{v}}{d \mathbf{m}} = \mathbf{C}  \mathbf{M^e_{mu^{-1}}} \mathbf{C^{\\top}} \\frac{d \mathbf{M^f_{\sigma^{-1}}}\mathbf{v} }{d \mathbf{m}}

        :param float freq: frequency
        :param numpy.ndarray u: solution vector (nF,)
        :param numpy.ndarray v: vector to take prodct with (nP,) or (nD,) for adjoint
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative of the system matrix times a vector (nP,) or adjoint (nD,)
        """

        MeMuI = self.MeMuI
        MfRho = self.MfRho
        C = self.mesh.edgeCurl
        MfRhoDeriv = self.MfRhoDeriv(u)

        if adjoint:
            if self._makeASymmetric is True:
                v = MfRho * v
            return MfRhoDeriv.T * (C * (MeMuI.T * (C.T * v)))

        if self._makeASymmetric is True:
            return MfRho.T * (C * ( MeMuI * (C.T * (MfRhoDeriv * v) )))
        return C * (MeMuI * (C.T * (MfRhoDeriv * v)))


    def getRHS(self, freq):
        """
        Right hand side for the system

        .. math ::

            \mathbf{RHS} = \mathbf{C} \mathbf{M_{\mu}^e}^{-1}\mathbf{s_m} -i\omega \mathbf{s_e}

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
        :param SimPEG.EM.FDEM.SrcFDEM.BaseSrc src: FDEM source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of rhs deriv with a vector
        """

        C = self.mesh.edgeCurl
        MeMuI = self.MeMuI
        s_mDeriv, s_eDeriv = src.evalDeriv(self, adjoint=adjoint)

        if adjoint:
            if self._makeASymmetric:
                MfRho = self.MfRho
                v = MfRho*v
            return s_mDeriv(MeMuI.T * (C.T * v)) - 1j * omega(freq) * s_eDeriv(v)

        else:
            RHSDeriv = C * (MeMuI * s_mDeriv(v)) - 1j * omega(freq) * s_eDeriv(v)

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

        \\left(\mathbf{C}^{\\top} \mathbf{M_{\\rho}^f} \mathbf{C} + i \omega \mathbf{M_{\mu}^e}\\right) \mathbf{h} = \mathbf{M^e} \mathbf{s_m} + \mathbf{C}^{\\top} \mathbf{M_{\\rho}^f} \mathbf{s_e}

    :param SimPEG.Mesh.BaseMesh.BaseMesh mesh: mesh
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
            \mathbf{A} = \mathbf{C}^{\\top} \mathbf{M_{\\rho}^f} \mathbf{C} + i \omega \mathbf{M_{\mu}^e}


        :param float freq: Frequency
        :rtype: scipy.sparse.csr_matrix
        :return: A

        """

        MeMu = self.MeMu
        MfRho = self.MfRho
        C = self.mesh.edgeCurl

        return C.T * (MfRho * C) + 1j*omega(freq)*MeMu

    def getADeriv(self, freq, u, v, adjoint=False):
        """
        Product of the derivative of our system matrix with respect to the model and a vector

        .. math::
            \\frac{\mathbf{A}(\mathbf{m}) \mathbf{v}}{d \mathbf{m}} = \mathbf{C}^{\\top}\\frac{d \mathbf{M^f_{\\rho}}\mathbf{v} }{d\mathbf{m}}

        :param float freq: frequency
        :param numpy.ndarray u: solution vector (nE,)
        :param numpy.ndarray v: vector to take prodct with (nP,) or (nD,) for adjoint
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative of the system matrix times a vector (nP,) or adjoint (nD,)
        """

        MeMu = self.MeMu
        C = self.mesh.edgeCurl
        MfRhoDeriv = self.MfRhoDeriv(C*u)

        if adjoint:
            return MfRhoDeriv.T * (C * v)
        return C.T * (MfRhoDeriv * v)

    def getRHS(self, freq):
        """
        Right hand side for the system

        .. math ::

            \mathbf{RHS} = \mathbf{M^e} \mathbf{s_m} + \mathbf{C}^{\\top} \mathbf{M_{\\rho}^f} \mathbf{s_e}

        :param float freq: Frequency
        :rtype: numpy.ndarray
        :return: RHS (nE, nSrc)

        """

        s_m, s_e = self.getSourceTerm(freq)
        C = self.mesh.edgeCurl
        MfRho  = self.MfRho

        return s_m + C.T * ( MfRho * s_e )

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model

        :param float freq: frequency
        :param SimPEG.EM.FDEM.SrcFDEM.BaseSrc src: FDEM source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of rhs deriv with a vector
        """

        _, s_e = src.eval(self)
        C = self.mesh.edgeCurl
        MfRho  = self.MfRho

        MfRhoDeriv = self.MfRhoDeriv(s_e)
        if not adjoint:
            RHSDeriv = C.T * (MfRhoDeriv * v)
        elif adjoint:
            RHSDeriv = MfRhoDeriv.T * (C * v)

        s_mDeriv, s_eDeriv = src.evalDeriv(self, adjoint=adjoint)

        return RHSDeriv + s_mDeriv(v) + C.T * (MfRho * s_eDeriv(v))

