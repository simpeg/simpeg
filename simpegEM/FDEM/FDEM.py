from SimPEG import Survey, Problem, Utils, np, sp, Solver as SimpegSolver
from scipy.constants import mu_0
from SurveyFDEM import SurveyFDEM
from FieldsFDEM import FieldsFDEM, FieldsFDEM_e, FieldsFDEM_b, FieldsFDEM_h, FieldsFDEM_j
from simpegEM.Base import BaseEMProblem
from simpegEM.Utils.EMUtils import omega


class BaseFDEMProblem(BaseEMProblem):
    """
        We start by looking at Maxwell's equations in the electric field \\(\\vec{E}\\) and the magnetic flux density \\(\\vec{B}\\):

        .. math::

            \\nabla \\times \\vec{E} + i \\omega \\vec{B} = \\vec{S_m} \\\\
            \\nabla \\times \\mu^{-1} \\vec{B} - \\sigma \\vec{E} = \\vec{S_e}

    """
    surveyPair = SurveyFDEM
    fieldsPair = FieldsFDEM

    def forward(self, m, RHS):

        F = self.fieldsPair(self.mesh, self.survey)

        for freq in self.survey.freqs:
            A = self.getA(freq)
            rhs = RHS(freq)
            Ainv = self.Solver(A, **self.solverOpts)
            sol = Ainv * rhs
            Srcs = self.survey.getSrcByFreq(freq)
            F[Srcs, self._fieldType] = sol

        return F

    def Jvec(self, m, v, u=None):
        if u is None:
           u = self.fields(m)

        self.curModel = m

        Jv = self.dataPair(self.survey)

        for freq in self.survey.freqs:
            A = self.getA(freq)
            Ainv = self.Solver(A, **self.solverOpts)

            for src in self.survey.getSource(freq):
                u_src = u[src, self.solType]
                w = self.getADeriv(freq, u_src, v)
                Ainvw = Ainv * w
                for rx in src.rxList:
                    fAinvw = self.calcFields(Ainvw, freq, rx.projField)
                    P = lambda v: rx.projectFieldsDeriv(src, self.mesh, u, v)

                    Jv[src, rx] = - P(fAinvw)

                    df_dm = self.calcFieldsDeriv(u_src, freq, rx.projField, v)
                    if df_dm is not None:
                        Jv[src, rx] += P(df_dm)

        return Utils.mkvc(Jv)

    def Jtvec(self, m, v, u=None):
        if u is None:
            u = self.fields(m)

        self.curModel = m

        # Ensure v is a data object.
        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        Jtv = np.zeros(self.mapping.nP)

        for freq in self.survey.freqs:
            AT = self.getA(freq).T
            ATinv = self.Solver(AT, **self.solverOpts)

            for src in self.survey.getSource(freq):
                u_src = u[src, self.solType]

                for rx in src.rxList:
                    PTv = rx.projectFieldsDeriv(src, self.mesh, u, v[src, rx], adjoint=True)
                    fPTv = self.calcFields(PTv, freq, rx.projField, adjoint=True)

                    w = ATinv * fPTv
                    Jtv_rx = - self.getADeriv(freq, u_src, w, adjoint=True)

                    df_dm = self.calcFieldsDeriv(u_src, freq, rx.projField, PTv, adjoint=True)

                    if df_dm is not None:
                        Jtv_rx += df_dm

                    real_or_imag = rx.projComp
                    if real_or_imag == 'real':
                        Jtv +=   Jtv_rx.real
                    elif real_or_imag == 'imag':
                        Jtv += - Jtv_rx.real
                    else:
                        raise Exception('Must be real or imag')

        return Jtv

    def getSourceTerm(self, freq):
        """
            :param float freq: Frequency
            :rtype: numpy.ndarray (nE or nF, nSrc)
            :return: RHS
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
                S_m[:,i] = smi
            if sei is not None:
                S_e[:,i] = sei

        return S_m, S_e

    def getSourceTermDeriv(self,freq,m,v,u=None,adjoint=False):
        raise NotImplementedError('getSourceTermDeriv not implemented yet')
        return None, None


##########################################################################################
################################ E-B Formulation #########################################
##########################################################################################

class ProblemFDEM_e(BaseFDEMProblem):
    """
        By eliminating the magnetic flux density using

        .. math::

            \\vec{B} = \\frac{-1}{i\\omega}\\nabla\\times\\vec{E},

        we can write Maxwell's equations as a second order system in \\ \\vec{E} \\ only:

        .. math::

            \\nabla \\times \\mu^{-1} \\nabla \\times \\vec{E} + i \\omega \\sigma \\vec{E} = \\vec{J_s}

        This is the definition of the Forward Problem using the E-formulation of Maxwell's equations.


    """

    _fieldType = 'e'
    _eqLocs    = 'FE'
    fieldsPair = FieldsFDEM_e

    def __init__(self, mesh, **kwargs):
        BaseFDEMProblem.__init__(self, mesh, **kwargs)

    def getA(self, freq):
        """
            :param float freq: Frequency
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """
        mui = self.MfMui
        sig = self.MeSigma
        C = self.mesh.edgeCurl

        return C.T*mui*C + 1j*omega(freq)*sig


    def getADeriv(self, freq, u, v, adjoint=False):
        sig = self.curModel.transform
        dsig_dm = self.curModel.transformDeriv
        dMe_dsig = self.mesh.getEdgeInnerProductDeriv(sig)(u)

        if adjoint:
            return 1j * omega(freq) * ( dsig_dm.T * ( dMe_dsig.T * v ) )

        return 1j * omega(freq) * ( dMe_dsig * ( dsig_dm * v ) )

    def getRHS(self, freq):
        """
            :param float freq: Frequency
            :rtype: numpy.ndarray (nE, nSrc)
            :return: RHS
        """

        S_m, S_e = self.getSourceTerm(freq)
        C = self.mesh.edgeCurl
        MfMui = self.MfMui

        RHS = C.T * (MfMui * S_m) -1j*omega(freq)*S_e

        return RHS

    def getRHSDeriv(self, freq, u, v, adjoint=False):
        raise NotImplementedError('getRHSDeriv not implemented yet')
        return None


class ProblemFDEM_b(BaseFDEMProblem):
    """
        Solving for b!
    """
    _fieldType = 'b'
    _eqLocs    = 'FE'
    fieldsPair = FieldsFDEM_b

    def __init__(self, mesh, **kwargs):
        BaseFDEMProblem.__init__(self, mesh, **kwargs)

    def getA(self, freq):
        """
            :param float freq: Frequency
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """
        mui = self.MfMui
        sigI = self.MeSigmaI
        C = self.mesh.edgeCurl
        iomega = 1j * omega(freq) * sp.eye(self.mesh.nF)

        A = C*sigI*C.T*mui + iomega

        if self._makeASymmetric is True:
            return mui.T*A
        return A

    def getADeriv(self, freq, u, v, adjoint=False):

        mui = self.MfMui
        C = self.mesh.edgeCurl
        sig = self.curModel.transform
        dsig_dm = self.curModel.transformDeriv
        dMeSigmaI_dI = self._dMeSigmaI_dI

        vec = (C.T*(mui*u))
        dMe_dsig = self.mesh.getEdgeInnerProductDeriv(sig)(vec)

        if adjoint:
            if self._makeASymmetric is True:
                v = mui * v
            return dsig_dm.T * ( dMe_dsig.T * ( dMeSigmaI_dI.T * ( C.T * v ) ) )

        if self._makeASymmetric is True:
            return mui.T * ( C * ( dMeSigmaI_dI * ( dMe_dsig * ( dsig_dm * v ) ) ) )
        return C * ( dMeSigmaI_dI * ( dMe_dsig * ( dsig_dm * v ) ) )


    def getRHS(self, freq):
        """
            :param float freq: Frequency
            :rtype: numpy.ndarray (nE, nSrc)
            :return: RHS
        """

        S_m, S_e = self.getSourceTerm(freq)
        C = self.mesh.edgeCurl
        MeSigmaI = self.MeSigmaI

        RHS = S_m + C * ( MeSigmaI * S_e )

        if self._makeASymmetric is True:
            mui = self.MfMui
            return mui.T*RHS

        return RHS

    def getRHSDeriv(self, freq, u, v, adjoint=False):
        raise NotImplementedError('getRHSDeriv not implemented yet')
        return None



##########################################################################################
################################ H-J Formulation #########################################
##########################################################################################


class ProblemFDEM_j(BaseFDEMProblem):
    """
        Using the H-J formulation of Maxwell's equations

        .. math::
            \\nabla \\times \\sigma^{-1} \\vec{J} + i\\omega\\mu\\vec{H} = 0
            \\nabla \\times \\vec{H} - \\vec{J} = \\vec{J_s}

        Since \(\\vec{J}\) is a flux and \(\\vec{H}\) is a field, we discretize \(\\vec{J}\) on faces and \(\\vec{H}\) on edges.

        For this implementation, we solve for J using \( \\vec{H} = - (i\\omega\\mu)^{-1} \\nabla \\times \\sigma^{-1} \\vec{J} \) :

        .. math::
            \\nabla \\times ( \\mu^{-1} \\nabla \\times \\sigma^{-1} \\vec{J} ) + i\\omega \\vec{J} = - i\\omega\\vec{J_s}

        We discretize this to:

        .. math::
            (\\mathbf{C}  \\mathbf{M^e_{mu^{-1}}} \\mathbf{C^T} \\mathbf{M^f_{\\sigma^{-1}}}  + i\\omega ) \\mathbf{j} = - i\\omega \\mathbf{j_s}

        .. note::
            This implementation does not yet work with full anisotropy!!

    """

    _fieldType = 'j'
    _eqLocs    = 'EF'
    fieldsPair = FieldsFDEM_j

    def __init__(self, mesh, **kwargs):
        BaseFDEMProblem.__init__(self, mesh, **kwargs)

    def getA(self, freq):
        """
            Here, we form the operator \(\\mathbf{A}\) to solce
            .. math::
                    \\mathbf{A} = \\mathbf{C}  \\mathbf{M^e_{mu^{-1}}} \\mathbf{C^T} \\mathbf{M^f_{\\sigma^{-1}}}  + i\\omega

            :param float freq: Frequency
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """

        MeMuI = self.MeMuI
        MfSigi = self.MfSigmai
        C = self.mesh.edgeCurl
        iomega = 1j * omega(freq) * sp.eye(self.mesh.nF)

        A = C * MeMuI * C.T * MfSigi + iomega

        if self._makeASymmetric is True:
            return MfSigi.T*A
        return A


    def getADeriv(self, freq, u, v, adjoint=False):
        """
            In this case, we assume that electrical conductivity, \(\\sigma\) is the physical property of interest (i.e. \(\sigma\) = model.transform). Then we want
            .. math::
                \\frac{\mathbf{A(\\sigma)} \mathbf{v}}{d \\mathbf{m}} &= \\mathbf{C}  \\mathbf{M^e_{mu^{-1}}} \\mathbf{C^T} \\frac{d \\mathbf{M^f_{\\sigma^{-1}}}}{d \\mathbf{m}}
                &= \\mathbf{C}  \\mathbf{M^e_{mu}^{-1}} \\mathbf{C^T} \\frac{d \\mathbf{M^f_{\\sigma^{-1}}}}{d \\mathbf{\\sigma^{-1}}} \\frac{d \\mathbf{\\sigma^{-1}}}{d \\mathbf{\\sigma}} \\frac{d \\mathbf{\\sigma}}{d \\mathbf{m}}
        """

        MeMuI = self.MeMuI
        MfSigi = self.MfSigmai
        C = self.mesh.edgeCurl
        sig = self.curModel.transform
        sigi = 1/sig
        dsig_dm = self.curModel.transformDeriv
        dsigi_dsig = -Utils.sdiag(sigi)**2
        dMf_dsigi = self.mesh.getFaceInnerProductDeriv(sigi)(u)

        if adjoint:
            if self._makeASymmetric is True:
                v = MfSigi * v
            return dsig_dm.T * ( dsigi_dsig.T *( dMf_dsigi.T * ( C * ( MeMuI.T * ( C.T * v ) ) ) ) )

        if self._makeASymmetric is True:
            return MfSigi.T * ( C * ( MeMuI * ( C.T * ( dMf_dsigi * ( dsigi_dsig * ( dsig_dm * v ) ) ) ) ) )
        return C * ( MeMuI * ( C.T * ( dMf_dsigi * ( dsigi_dsig * ( dsig_dm * v ) ) ) ) )


    def getRHS(self, freq):
        """
            :param float freq: Frequency
            :rtype: numpy.ndarray (nE, nSrc)
            :return: RHS
        """

        S_m, S_e = self.getSourceTerm(freq)
        C = self.mesh.edgeCurl
        MeMuI = self.MeMuI   


        RHS = C * (MeMuI * S_m) - 1j * omega(freq) * S_e
        if self._makeASymmetric is True:
            MfSigi = self.MfSigmai
            return MfSigi.T*RHS

        return RHS

    def getRHSDeriv(self, freq, u, v, adjoint=False):
        raise NotImplementedError('getRHSDeriv not implemented yet')
        return None



class ProblemFDEM_h(BaseFDEMProblem):
    """
        Using the H-J formulation of Maxwell's equations

        .. math::
            \\nabla \\times \\sigma^{-1} \\vec{J} + i\\omega\\mu\\vec{H} = 0
            \\nabla \\times \\vec{H} - \\vec{J} = \\vec{J_s}

        Since \(\\vec{J}\) is a flux and \(\\vec{H}\) is a field, we discretize \(\\vec{J}\) on faces and \(\\vec{H}\) on edges.

        For this implementation, we solve for J using \( \\vec{J} =  \\nabla \\times \\vec{H} - \\vec{J_s} \)

        .. math::
            \\nabla \\times \\sigma^{-1} \\nabla \\times \\vec{H} + i\\omega\\mu\\vec{H} = \\nabla \\times \\sigma^{-1} \\vec{J_s}

        We discretize and solve

        .. math::
            (\\mathbf{C^T} \\mathbf{M^f_{\\sigma^{-1}}} \\mathbf{C} + i\\omega \\mathbf{M_{\mu}} ) \\mathbf{h} = \\mathbf{C^T} \\mathbf{M^f_{\\sigma^{-1}}} \\vec{J_s}

        .. note::
            This implementation does not yet work with full anisotropy!!

    """

    _fieldType = 'h'
    _eqLocs    = 'EF'
    fieldsPair = FieldsFDEM_h

    def __init__(self, mesh, **kwargs):
        BaseFDEMProblem.__init__(self, mesh, **kwargs)

    def getA(self, freq):
        """
            :param float freq: Frequency
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """

        MeMu = self.MeMu
        MfSigi = self.MfSigmai
        C = self.mesh.edgeCurl

        return C.T * MfSigi * C + 1j*omega(freq)*MeMu

    def getADeriv(self, freq, u, v, adjoint=False):

        MeMu = self.MeMu
        C = self.mesh.edgeCurl
        sig = self.curModel.transform
        sigi = 1/sig
        dsig_dm = self.curModel.transformDeriv
        dsigi_dsig = -Utils.sdiag(sigi)**2

        dMf_dsigi = self.mesh.getFaceInnerProductDeriv(sigi)(C*u)

        if adjoint:
            return (dsig_dm.T * (dsigi_dsig.T * (dMf_dsigi.T * (C * v))))
        return  (C.T  * (dMf_dsigi * (dsigi_dsig * (dsig_dm * v))))


    def getRHS(self, freq):
        """
            :param float freq: Frequency
            :rtype: numpy.ndarray (nE, nSrc)
            :return: RHS
        """

        S_m, S_e = self.getSourceTerm(freq)
        C = self.mesh.edgeCurl
        MfSigmai  = self.MfSigmai

        RHS = S_m + C.T * ( MfSigmai * S_e )

        return RHS

    def getRHSDeriv(self, freq, u, v, adjoint=False):
        raise NotImplementedError('getRHSDeriv not implemented yet')
        return None

