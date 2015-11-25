from SimPEG import Problem, Utils, np, sp, Solver as SimpegSolver
from scipy.constants import mu_0
from SurveyFDEM import Survey as SurveyFDEM
from FieldsFDEM import Fields, Fields_e, Fields_b, Fields_h, Fields_j
from SimPEG.EM.Base import BaseEMProblem
from SimPEG.EM.Utils import omega


class BaseFDEMProblem(BaseEMProblem):
    """
        We start by looking at Maxwell's equations in the electric
        field \\\(\\\mathbf{e}\\\) and the magnetic flux
        density \\\(\\\mathbf{b}\\\)

        .. math ::

            \mathbf{C} \mathbf{e} + i \omega \mathbf{b} = \mathbf{s_m} \\\\
            {\mathbf{C}^T \mathbf{M_{\mu^{-1}}^f} \mathbf{b} - \mathbf{M_{\sigma}^e} \mathbf{e} = \mathbf{M^e} \mathbf{s_e}}

        if using the E-B formulation (:code:`Problem_e`
        or :code:`Problem_b`) or the magnetic field
        \\\(\\\mathbf{h}\\\) and current density \\\(\\\mathbf{j}\\\)

        .. math ::

            \mathbf{C}^T \mathbf{M_{\\rho}^f} \mathbf{j} + i \omega \mathbf{M_{\mu}^e} \mathbf{h} = \mathbf{M^e} \mathbf{s_m} \\\\
            \mathbf{C} \mathbf{h} - \mathbf{j} = \mathbf{s_e}

        if using the H-J formulation (:code:`Problem_j` or :code:`Problem_h`).

        The problem performs the elimination so that we are solving the system for \\\(\\\mathbf{e},\\\mathbf{b},\\\mathbf{j} \\\) or \\\(\\\mathbf{h}\\\)
    """

    surveyPair = SurveyFDEM
    fieldsPair = Fields

    def fields(self, m=None):
        """
            Solve the forward problem for the fields.
        """

        self.curModel = m
        F = self.fieldsPair(self.mesh, self.survey)

        for freq in self.survey.freqs:
            A = self.getA(freq)
            rhs = self.getRHS(freq)
            Ainv = self.Solver(A, **self.solverOpts)
            sol = Ainv * rhs
            Srcs = self.survey.getSrcByFreq(freq)
            ftype = self._fieldType + 'Solution'
            F[Srcs, ftype] = sol

        return F

    def Jvec(self, m, v, f=None):
        """
            Sensitivity times a vector
        """

        if f is None:
           f = self.fields(m)

        self.curModel = m

        Jv = self.dataPair(self.survey)

        for freq in self.survey.freqs:
            dA_du = self.getA(freq) #
            dA_duI = self.Solver(dA_du, **self.solverOpts)

            for src in self.survey.getSrcByFreq(freq):
                ftype = self._fieldType + 'Solution'
                u_src = f[src, ftype]
                dA_dm = self.getADeriv_m(freq, u_src, v)
                dRHS_dm = self.getRHSDeriv_m(src, v)
                if dRHS_dm is None:
                    du_dm = dA_duI * ( - dA_dm )
                else:
                    du_dm = dA_duI * ( - dA_dm + dRHS_dm )
                for rx in src.rxList:
                    # df_duFun = u.deriv_u(rx.fieldsUsed, m)
                    df_duFun = getattr(f, '_%sDeriv_u'%rx.projField, None)
                    df_du = df_duFun(src, du_dm, adjoint=False)
                    if df_du is not None:
                        du_dm = df_du

                    df_dmFun = getattr(f, '_%sDeriv_m'%rx.projField, None)
                    df_dm = df_dmFun(src, v, adjoint=False)
                    if df_dm is not None:
                        du_dm += df_dm

                    P = lambda v: rx.projectFieldsDeriv(src, self.mesh, f, v) # wrt u, also have wrt m


                    Jv[src, rx] = P(du_dm)

        return Utils.mkvc(Jv)

    def Jtvec(self, m, v, f=None):
        """
            Sensitivity transpose times a vector
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
                ftype = self._fieldType + 'Solution'
                u_src = f[src, ftype]

                for rx in src.rxList:
                    PTv = rx.projectFieldsDeriv(src, self.mesh, f, v[src, rx], adjoint=True) # wrt u, need possibility wrt m

                    df_duTFun = getattr(f, '_%sDeriv_u'%rx.projField, None)
                    df_duT = df_duTFun(src, PTv, adjoint=True)
                    if df_duT is not None:
                        dA_duIT = ATinv * df_duT
                    else:
                        dA_duIT = ATinv * PTv

                    dA_dmT = self.getADeriv_m(freq, u_src, dA_duIT, adjoint=True)

                    dRHS_dmT = self.getRHSDeriv_m(src, dA_duIT, adjoint=True)

                    if dRHS_dmT is None:
                        du_dmT = - dA_dmT
                    else:
                        du_dmT = -dA_dmT + dRHS_dmT

                    df_dmFun = getattr(f, '_%sDeriv_m'%rx.projField, None)
                    dfT_dm = df_dmFun(src, PTv, adjoint=True)
                    if dfT_dm is not None:
                        du_dmT += dfT_dm

                    real_or_imag = rx.projComp
                    if real_or_imag == 'real':
                        Jtv +=   du_dmT.real
                    elif real_or_imag == 'imag':
                        Jtv += - du_dmT.real
                    else:
                        raise Exception('Must be real or imag')

        return Jtv

    def getSourceTerm(self, freq):
        """
            Evaluates the sources for a given frequency and puts them in matrix form

            :param float freq: Frequency
            :rtype: numpy.ndarray (nE or nF, nSrc)
            :return: S_m, S_e
        """
        Srcs = self.survey.getSrcByFreq(freq)
        if self._eqLocs is 'FE':
            S_m = np.zeros((self.mesh.nF,len(Srcs)), dtype=complex)
            S_e = np.zeros((self.mesh.nE,len(Srcs)), dtype=complex)
        elif self._eqLocs is 'EF':
            S_m = np.zeros((self.mesh.nE,len(Srcs)), dtype=complex)
            S_e = np.zeros((self.mesh.nF,len(Srcs)), dtype=complex)

        for i, src in enumerate(Srcs):
            smi, sei = src.eval(self)
            if smi is not None:
                S_m[:,i] = Utils.mkvc(smi)
            if sei is not None:
                S_e[:,i] = Utils.mkvc(sei)

        return S_m, S_e


##########################################################################################
################################ E-B Formulation #########################################
##########################################################################################

class Problem_e(BaseFDEMProblem):
    """
        By eliminating the magnetic flux density using

            .. math ::

                \mathbf{b} = \\frac{1}{i \omega}\\left(-\mathbf{C} \mathbf{e} + \mathbf{s_m}\\right)


        we can write Maxwell's equations as a second order system in \\\(\\\mathbf{e}\\\) only:

        .. math ::

            \\left(\mathbf{C}^T \mathbf{M_{\mu^{-1}}^f} \mathbf{C}+ i \omega \mathbf{M^e_{\sigma}} \\right)\mathbf{e} = \mathbf{C}^T \mathbf{M_{\mu^{-1}}^f}\mathbf{s_m} -i\omega\mathbf{M^e}\mathbf{s_e}

        which we solve for \\\(\\\mathbf{e}\\\).
    """

    _fieldType = 'e'
    _eqLocs    = 'FE'
    fieldsPair = Fields_e

    def __init__(self, mesh, **kwargs):
        BaseFDEMProblem.__init__(self, mesh, **kwargs)

    def getA(self, freq):
        """
            .. math ::
                \mathbf{A} = \mathbf{C}^T \mathbf{M_{\mu^{-1}}^f} \mathbf{C} + i \omega \mathbf{M^e_{\sigma}}

            :param float freq: Frequency
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """
        MfMui = self.MfMui
        MeSigma = self.MeSigma
        C = self.mesh.edgeCurl

        return C.T*MfMui*C + 1j*omega(freq)*MeSigma


    def getADeriv_m(self, freq, u, v, adjoint=False):
        dsig_dm = self.curModel.sigmaDeriv
        dMe_dsig = self.MeSigmaDeriv(u)

        if adjoint:
            return 1j * omega(freq) * ( dMe_dsig.T * v )

        return 1j * omega(freq) * ( dMe_dsig * v )

    def getRHS(self, freq):
        """
            .. math ::
                \mathbf{RHS} = \mathbf{C}^T \mathbf{M_{\mu^{-1}}^f}\mathbf{s_m} -i\omega\mathbf{M_e}\mathbf{s_e}

            :param float freq: Frequency
            :rtype: numpy.ndarray (nE, nSrc)
            :return: RHS
        """

        S_m, S_e = self.getSourceTerm(freq)
        C = self.mesh.edgeCurl
        MfMui = self.MfMui

        # RHS = C.T * (MfMui * S_m) -1j * omega(freq) * Me * S_e
        RHS = C.T * (MfMui * S_m) -1j * omega(freq) * S_e

        return RHS

    def getRHSDeriv_m(self, src, v, adjoint=False):
        C = self.mesh.edgeCurl
        MfMui = self.MfMui
        S_mDeriv, S_eDeriv = src.evalDeriv(self, adjoint)

        if adjoint:
            dRHS = MfMui * (C * v)
            S_mDerivv = S_mDeriv(dRHS)
            S_eDerivv = S_eDeriv(v)
            if S_mDerivv is not None and S_eDerivv is not None:
                return S_mDerivv - 1j * omega(freq) * S_eDerivv
            elif S_mDerivv is not None:
                return S_mDerivv
            elif S_eDerivv is not None:
                return - 1j * omega(freq) * S_eDerivv
            else:
                return None
        else:
            S_mDerivv, S_eDerivv = S_mDeriv(v), S_eDeriv(v)

            if S_mDerivv is not None and S_eDerivv is not None:
                return C.T * (MfMui * S_mDerivv) -1j * omega(freq) * S_eDerivv
            elif S_mDerivv is not None:
                return C.T * (MfMui * S_mDerivv)
            elif S_eDerivv is not None:
                return -1j * omega(freq) * S_eDerivv
            else:
                return None


class Problem_b(BaseFDEMProblem):
    """
        We eliminate \\\(\\\mathbf{e}\\\) using

        .. math ::

             \mathbf{e} = \mathbf{M^e_{\sigma}}^{-1} \\left(\mathbf{C}^T \mathbf{M_{\mu^{-1}}^f} \mathbf{b} - \mathbf{s_e}\\right)

        and solve for \\\(\\\mathbf{b}\\\) using:

        .. math ::

            \\left(\mathbf{C} \mathbf{M^e_{\sigma}}^{-1} \mathbf{C}^T \mathbf{M_{\mu^{-1}}^f}  + i \omega \\right)\mathbf{b} = \mathbf{s_m} + \mathbf{M^e_{\sigma}}^{-1}\mathbf{M^e}\mathbf{s_e}

        .. note ::
            The inverse problem will not work with full anisotropy
    """

    _fieldType = 'b'
    _eqLocs    = 'FE'
    fieldsPair = Fields_b

    def __init__(self, mesh, **kwargs):
        BaseFDEMProblem.__init__(self, mesh, **kwargs)

    def getA(self, freq):
        """
            .. math ::
                \mathbf{A} = \mathbf{C} \mathbf{M^e_{\sigma}}^{-1} \mathbf{C}^T \mathbf{M_{\mu^{-1}}^f}  + i \omega

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

    def getADeriv_m(self, freq, u, v, adjoint=False):

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
            .. math ::
                \mathbf{RHS} = \mathbf{s_m} + \mathbf{M^e_{\sigma}}^{-1}\mathbf{s_e}

            :param float freq: Frequency
            :rtype: numpy.ndarray (nE, nSrc)
            :return: RHS
        """

        S_m, S_e = self.getSourceTerm(freq)
        C = self.mesh.edgeCurl
        MeSigmaI = self.MeSigmaI
        # Me = self.Me

        RHS = S_m + C * ( MeSigmaI * S_e )

        if self._makeASymmetric is True:
            MfMui = self.MfMui
            return MfMui.T * RHS

        return RHS

    def getRHSDeriv_m(self, src, v, adjoint=False):
        C = self.mesh.edgeCurl
        S_m, S_e = src.eval(self)
        MfMui = self.MfMui
        # Me = self.Me

        if self._makeASymmetric and adjoint:
            v = self.MfMui * v

        if S_e is not None:
            MeSigmaIDeriv = self.MeSigmaIDeriv(S_e)
            if not adjoint:
                RHSderiv = C * (MeSigmaIDeriv * v)
            elif adjoint:
                RHSderiv = MeSigmaIDeriv.T * (C.T * v)
        else:
            RHSderiv = None

        S_mDeriv, S_eDeriv = src.evalDeriv(self, adjoint)
        S_mDeriv, S_eDeriv = S_mDeriv(v), S_eDeriv(v)
        if S_mDeriv is not None and S_eDeriv is not None:
            if not adjoint:
                SrcDeriv = S_mDeriv + C * (self.MeSigmaI * S_eDeriv)
            elif adjoint:
                SrcDeriv = S_mDeriv + Self.MeSigmaI.T * ( C.T * S_eDeriv)
        elif S_mDeriv is not None:
            SrcDeriv = S_mDeriv
        elif S_eDeriv is not None:
            if not adjoint:
                SrcDeriv = C * (self.MeSigmaI * S_eDeriv)
            elif adjoint:
                SrcDeriv = self.MeSigmaI.T * ( C.T * S_eDeriv)
        else:
            SrcDeriv = None

        if RHSderiv is not None and SrcDeriv is not None:
            RHSderiv += SrcDeriv
        elif SrcDeriv is not None:
            RHSderiv = SrcDeriv

        if RHSderiv is not None:
            if self._makeASymmetric is True and not adjoint:
                return MfMui.T * RHSderiv

        return RHSderiv



##########################################################################################
################################ H-J Formulation #########################################
##########################################################################################


class Problem_j(BaseFDEMProblem):
    """
        We eliminate \\\(\\\mathbf{h}\\\) using

        .. math ::

            \mathbf{h} = \\frac{1}{i \omega} \mathbf{M_{\mu}^e}^{-1} \\left(-\mathbf{C}^T \mathbf{M_{\\rho}^f} \mathbf{j}  + \mathbf{M^e} \mathbf{s_m} \\right)

        and solve for \\\(\\\mathbf{j}\\\) using

        .. math ::

            \\left(\mathbf{C} \mathbf{M_{\mu}^e}^{-1} \mathbf{C}^T \mathbf{M_{\\rho}^f} + i \omega\\right)\mathbf{j} = \mathbf{C} \mathbf{M_{\mu}^e}^{-1} \mathbf{M^e} \mathbf{s_m} -i\omega\mathbf{s_e}

        .. note::
            This implementation does not yet work with full anisotropy!!

    """

    _fieldType = 'j'
    _eqLocs    = 'EF'
    fieldsPair = Fields_j

    def __init__(self, mesh, **kwargs):
        BaseFDEMProblem.__init__(self, mesh, **kwargs)

    def getA(self, freq):
        """
            .. math ::
                    \\mathbf{A} = \\mathbf{C}  \\mathbf{M^e_{mu^{-1}}} \\mathbf{C}^T \\mathbf{M^f_{\\sigma^{-1}}}  + i\\omega

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


    def getADeriv_m(self, freq, u, v, adjoint=False):
        """
            In this case, we assume that electrical conductivity, \\\(\\\sigma\\\) is the physical property of interest (i.e. \\\(\\\sigma\\\) = model.transform). Then we want

            .. math ::

                \\frac{\mathbf{A(\sigma)} \mathbf{v}}{d \\mathbf{m}} &= \\mathbf{C}  \\mathbf{M^e_{mu^{-1}}} \\mathbf{C^T} \\frac{d \\mathbf{M^f_{\\sigma^{-1}}}}{d \\mathbf{m}}
                &= \\mathbf{C}  \\mathbf{M^e_{mu}^{-1}} \\mathbf{C^T} \\frac{d \\mathbf{M^f_{\\sigma^{-1}}}}{d \\mathbf{\\sigma^{-1}}} \\frac{d \\mathbf{\\sigma^{-1}}}{d \\mathbf{\\sigma}} \\frac{d \\mathbf{\\sigma}}{d \\mathbf{m}}
        """

        MeMuI = self.MeMuI
        MfRho = self.MfRho
        C = self.mesh.edgeCurl
        MfRhoDeriv_m = self.MfRhoDeriv(u)

        if adjoint:
            if self._makeASymmetric is True:
                v = MfRho * v
            return MfRhoDeriv_m.T * (C * (MeMuI.T * (C.T * v)))

        if self._makeASymmetric is True:
            return MfRho.T * (C * ( MeMuI * (C.T * (MfRhoDeriv_m * v) )))
        return C * (MeMuI * (C.T * (MfRhoDeriv_m * v)))


    def getRHS(self, freq):
        """
            .. math ::

                \mathbf{RHS} = \mathbf{C} \mathbf{M_{\mu}^e}^{-1}\mathbf{s_m} -i\omega \mathbf{s_e}
            :param float freq: Frequency
            :rtype: numpy.ndarray (nE, nSrc)
            :return: RHS
        """

        S_m, S_e = self.getSourceTerm(freq)
        C = self.mesh.edgeCurl
        MeMuI = self.MeMuI

        RHS = C * (MeMuI * S_m) - 1j * omega(freq) * S_e
        if self._makeASymmetric is True:
            MfRho = self.MfRho
            return MfRho.T*RHS

        return RHS

    def getRHSDeriv_m(self, src, v, adjoint=False):
        C = self.mesh.edgeCurl
        MeMuI = self.MeMuI
        S_mDeriv, S_eDeriv = src.evalDeriv(self, adjoint)

        if adjoint:
            if self._makeASymmetric:
                MfRho = self.MfRho
                v = MfRho*v
            S_mDerivv = S_mDeriv(MeMuI.T * (C.T * v))
            S_eDerivv = S_eDeriv(v)
            if S_mDerivv is not None and S_eDerivv is not None:
                return S_mDerivv - 1j * omega(freq) * S_eDerivv
            elif S_mDerivv is not None:
                return S_mDerivv
            elif S_eDerivv is not None:
                return - 1j * omega(freq) * S_eDerivv
            else:
                return None
        else:
            S_mDerivv, S_eDerivv = S_mDeriv(v), S_eDeriv(v)

            if S_mDerivv is not None and S_eDerivv is not None:
                RHSDeriv = C * (MeMuI * S_mDerivv) - 1j * omega(freq) * S_eDerivv
            elif S_mDerivv is not None:
                RHSDeriv = C * (MeMuI * S_mDerivv)
            elif S_eDerivv is not None:
                RHSDeriv = - 1j * omega(freq) * S_eDerivv
            else:
                return None

            if self._makeASymmetric:
                MfRho = self.MfRho
                return MfRho.T * RHSDeriv
            return RHSDeriv




class Problem_h(BaseFDEMProblem):
    """
        We eliminate \\\(\\\mathbf{j}\\\) using

        .. math ::

            \mathbf{j} = \mathbf{C} \mathbf{h} - \mathbf{s_e}

        and solve for \\\(\\\mathbf{h}\\\) using

        .. math ::

            \\left(\mathbf{C}^T \mathbf{M_{\\rho}^f} \mathbf{C} + i \omega \mathbf{M_{\mu}^e}\\right) \mathbf{h} = \mathbf{M^e} \mathbf{s_m} + \mathbf{C}^T \mathbf{M_{\\rho}^f} \mathbf{s_e}

    """

    _fieldType = 'h'
    _eqLocs    = 'EF'
    fieldsPair = Fields_h

    def __init__(self, mesh, **kwargs):
        BaseFDEMProblem.__init__(self, mesh, **kwargs)

    def getA(self, freq):
        """
            .. math ::

                \mathbf{A} = \mathbf{C}^T \mathbf{M_{\\rho}^f} \mathbf{C} + i \omega \mathbf{M_{\mu}^e}

            :param float freq: Frequency
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """

        MeMu = self.MeMu
        MfRho = self.MfRho
        C = self.mesh.edgeCurl

        return C.T * (MfRho * C) + 1j*omega(freq)*MeMu

    def getADeriv_m(self, freq, u, v, adjoint=False):

        MeMu = self.MeMu
        C = self.mesh.edgeCurl
        MfRhoDeriv_m = self.MfRhoDeriv(C*u)

        if adjoint:
            return MfRhoDeriv_m.T * (C * v)
        return C.T * (MfRhoDeriv_m * v)

    def getRHS(self, freq):
        """
            .. math ::

                \mathbf{RHS} = \mathbf{M^e} \mathbf{s_m} + \mathbf{C}^T \mathbf{M_{\\rho}^f} \mathbf{s_e}

            :param float freq: Frequency
            :rtype: numpy.ndarray (nE, nSrc)
            :return: RHS
        """

        S_m, S_e = self.getSourceTerm(freq)
        C = self.mesh.edgeCurl
        MfRho  = self.MfRho

        RHS = S_m + C.T * ( MfRho * S_e )

        return RHS

    def getRHSDeriv_m(self, src, v, adjoint=False):
        _, S_e = src.eval(self)
        C = self.mesh.edgeCurl
        MfRho  = self.MfRho

        RHSDeriv = None

        if S_e is not None:
            MfRhoDeriv = self.MfRhoDeriv(S_e)
            if not adjoint:
                RHSDeriv = C.T * (MfRhoDeriv * v)
            elif adjoint:
                RHSDeriv = MfRhoDeriv.T * (C * v)

        S_mDeriv, S_eDeriv = src.evalDeriv(self, adjoint)

        S_mDeriv = S_mDeriv(v)
        S_eDeriv = S_eDeriv(v)

        if S_mDeriv is not None:
            if RHSDeriv is not None:
                RHSDeriv += S_mDeriv(v)
            else:
                RHSDeriv =  S_mDeriv(v)
        if S_eDeriv is not None:
            if RHSDeriv is not None:
                RHSDeriv += C.T * (MfRho * S_e)
            else:
                RHSDeriv = C.T * (MfRho * S_e)

        return RHSDeriv

