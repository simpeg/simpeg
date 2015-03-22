from SimPEG import Survey, Problem, Utils, np, sp, Solver as SimpegSolver
from scipy.constants import mu_0
from SurveyFDEM import SurveyFDEM, FieldsFDEM
from simpegEM import Sources
from simpegEM.Base import BaseEMProblem

def omega(freq):
    """Change frequency to angular frequency, omega"""
    return 2.*np.pi*freq

def getSource(self,freq):
        """
            :param float freq: Frequency
            :rtype: numpy.ndarray (nE, nTx)
            :return: RHS
        """
        Txs = self.survey.getTransmitters(freq)
        rhs = range(len(Txs))

        solType = self.solType

        if solType == 'e' or solType == 'b':
            gridEJx = self.mesh.gridEx
            gridEJy = self.mesh.gridEy
            gridEJz = self.mesh.gridEz
            nEJ = self.mesh.nE

            gridBHx = self.mesh.gridFx
            gridBHy = self.mesh.gridFy
            gridBHz = self.mesh.gridFz
            nBH = self.mesh.nF


            C = self.mesh.edgeCurl
            mui = self.MfMui

        elif solType == 'h' or solType == 'j':
            gridEJx = self.mesh.gridFx
            gridEJy = self.mesh.gridFy
            gridEJz = self.mesh.gridFz
            nEJ = self.mesh.nF

            gridBHx = self.mesh.gridEx
            gridBHy = self.mesh.gridEy
            gridBHz = self.mesh.gridEz
            nBH = self.mesh.nE

            C = self.mesh.edgeCurl.T
            mui = self.MeMuI

        else:
            NotImplementedError('Only E or F sources')

        for i, tx in enumerate(Txs):
            if self.mesh._meshType is 'CYL':
                if self.mesh.isSymmetric:
                    if tx.txType == 'VMD':
                        SRC = Sources.MagneticDipoleVectorPotential(tx.loc, gridEJy, 'y')
                    elif tx.txType =='CircularLoop':
                        SRC = Sources.MagneticLoopVectorPotential(tx.loc, gridEJy, 'y', tx.radius)
                    else:
                        raise NotImplementedError('Only VMD and CircularLoop')
                else:
                    raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')

            elif self.mesh._meshType is 'TENSOR':

                if tx.txType == 'VMD':
                    src = Sources.MagneticDipoleVectorPotential
                    SRCx = src(tx.loc, gridEJx, 'x')
                    SRCy = src(tx.loc, gridEJy, 'y')
                    SRCz = src(tx.loc, gridEJz, 'z')

                elif tx.txType == 'VMD_B':
                    src = Sources.MagneticDipoleFields
                    SRCx = src(tx.loc, gridBHx, 'x')
                    SRCy = src(tx.loc, gridBHy, 'y')
                    SRCz = src(tx.loc, gridBHz, 'z')

                elif tx.txType == 'CircularLoop':
                    src = Sources.MagneticLoopVectorPotential
                    SRCx = src(tx.loc, gridEJx, 'x', tx.radius)
                    SRCy = src(tx.loc, gridEJy, 'y', tx.radius)
                    SRCz = src(tx.loc, gridEJz, 'z', tx.radius)
                else:

                    raise NotImplemented('%s txType is not implemented' % tx.txType)
                SRC = np.concatenate((SRCx, SRCy, SRCz))

            else:
                raise Exception('Unknown mesh for VMD')

            rhs[i] = SRC

        # b-forumlation
        if tx.txType == 'VMD_B':
            b_0 = np.concatenate(rhs).reshape((nBH, len(Txs)), order='F')
        else:
            a = np.concatenate(rhs).reshape((nEJ, len(Txs)), order='F')
            b_0 = C*a

        if solType == 'b' or solType == 'h':
            return b_0
        elif solType == 'e' or solType == 'j':
            return C.T*mui*b_0

class BaseFDEMProblem(BaseEMProblem):
    """
        We start by looking at Maxwell's equations in the electric field \\(\\vec{E}\\) and the magnetic flux density \\(\\vec{B}\\):

        .. math::

            \\nabla \\times \\vec{E} + i \\omega \\vec{B} = 0 \\\\
            \\nabla \\times \\mu^{-1} \\vec{B} - \\sigma \\vec{E} = \\vec{J_s}

    """
    surveyPair = SurveyFDEM

    def forward(self, m, RHS, CalcFields):

        F = FieldsFDEM(self.mesh, self.survey)

        for freq in self.survey.freqs:
            A = self.getA(freq)
            rhs = RHS(freq)
            Ainv = self.Solver(A, **self.solverOpts)
            sol = Ainv * rhs
            for fieldType in self.storeTheseFields:
                Txs = self.survey.getTransmitters(freq)
                F[Txs, fieldType] = CalcFields(sol, freq, fieldType)

        return F

    def Jvec(self, m, v, u=None):
        if u is None:
           u = self.fields(m)

        self.curModel = m

        Jv = self.dataPair(self.survey)

        for freq in self.survey.freqs:
            A = self.getA(freq)
            Ainv = self.Solver(A, **self.solverOpts)

            for tx in self.survey.getTransmitters(freq):
                u_tx = u[tx, self.solType]
                w = self.getADeriv(freq, u_tx, v)
                Ainvw = Ainv * w
                for rx in tx.rxList:
                    fAinvw = self.calcFields(Ainvw, freq, rx.projField)
                    P = lambda v: rx.projectFieldsDeriv(tx, self.mesh, u, v)

                    df_dm = self.calcFieldsDeriv(u_tx, freq, rx.projField, v)
                    if df_dm is None:
                        Jv[tx, rx] = - P(fAinvw)
                    else:
                        Jv[tx, rx] = - P(fAinvw) + P(df_dm)

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

            for tx in self.survey.getTransmitters(freq):
                u_tx = u[tx, self.solType]

                for rx in tx.rxList:
                    PTv = rx.projectFieldsDeriv(tx, self.mesh, u, v[tx, rx], adjoint=True)
                    fPTv = self.calcFields(PTv, freq, rx.projField, adjoint=True)

                    w = ATinv * fPTv
                    Jtv_rx = - self.getADeriv(freq, u_tx, w, adjoint=True)

                    df_dm = self.calcFieldsDeriv(u_tx, freq, rx.projField, PTv, adjoint=True)

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

    def getSource(self,freq):
        return self.getSource(freq)

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
    solType = 'e'

    def __init__(self, model, **kwargs):
        BaseFDEMProblem.__init__(self, model, **kwargs)

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
            :rtype: numpy.ndarray (nE, nTx)
            :return: RHS
        """

        j_s = getSource(self,freq)
        return -1j*omega(freq)*j_s

    def calcFields(self, sol, freq, fieldType, adjoint=False):
        e = sol
        if fieldType == 'e':
            return e
        elif fieldType == 'b':
            if not adjoint:
                b = -(1./(1j*omega(freq))) * ( self.mesh.edgeCurl * e )
            else:
                b = -(1./(1j*omega(freq))) * ( self.mesh.edgeCurl.T * e )
            return b
        raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)

    def calcFieldsDeriv(self, sol, freq, fieldType, v, adjoint=False):
        e = sol
        if fieldType == 'e':
            return None
        elif fieldType == 'b':
            return None
        raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)


class ProblemFDEM_b(BaseFDEMProblem):
    """
        Solving for b!
    """
    solType = 'b'

    def __init__(self, model, **kwargs):
        BaseFDEMProblem.__init__(self, model, **kwargs)

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
        #TODO: This only works if diagonal (no tensors)...
        dMeSigmaI_dI = - self.MeSigmaI**2

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
            :rtype: numpy.ndarray (nE, nTx)
            :return: RHS
        """

        b_0 = getSource(self,freq)

        rhs = -1j*omega(freq)*b_0
        if self._makeASymmetric is True:
            mui = self.MfMui
            return mui.T*rhs
        return rhs

    def calcFields(self, sol, freq, fieldType, adjoint=False):
        b = sol
        if fieldType == 'e':
            if not adjoint:
                e = self.MeSigmaI * ( self.mesh.edgeCurl.T * ( self.MfMui * b ) )
            else:
                e = self.MfMui.T * ( self.mesh.edgeCurl * ( self.MeSigmaI.T * b ) )
            return e
        elif fieldType == 'b':
            return b
        raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)

    def calcFieldsDeriv(self, sol, freq, fieldType, v, adjoint=False):
        b = sol
        if fieldType == 'e':
            sig = self.curModel.transform
            dsig_dm = self.curModel.transformDeriv

            C = self.mesh.edgeCurl
            mui = self.MfMui

            #TODO: This only works if diagonal (no tensors)...
            dMeSigmaI_dI = - self.MeSigmaI**2

            vec = C.T * ( mui * b )
            dMe_dsig = self.mesh.getEdgeInnerProductDeriv(sig)(vec)
            if not adjoint:
                return dMeSigmaI_dI * ( dMe_dsig * ( dsig_dm * v ) )
            else:
                return dsig_dm.T * ( dMe_dsig.T * ( dMeSigmaI_dI.T * v ) )
        elif fieldType == 'b':
            return None
        raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)


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

    solType = 'j'
    storeTheseFields = ['j','h']

    def __init__(self, model, **kwargs):
        BaseFDEMProblem.__init__(self, model, **kwargs)

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
            :rtype: numpy.ndarray (nE, nTx)
            :return: RHS
        """
        j_s = getSource(self,freq)
        rhs = -1j*omega(freq)*j_s

        if self._makeASymmetric is True:
            MfSigi = self.MfSigmai
            return MfSigi.T*rhs
        return rhs

    def calcFields(self, sol, freq, fieldType, adjoint=False):
        j = sol
        if fieldType == 'j':
            return j
        elif fieldType == 'h':
            MeMuI = self.MeMuI
            C = self.mesh.edgeCurl
            MfSigi = self.MfSigmai
            if not adjoint:
                h = -(1./(1j*omega(freq))) * MeMuI * ( C.T * ( MfSigi * j ) )
            else:
                h = -(1./(1j*omega(freq))) * MfSigi.T * ( C * ( MeMuI.T * j ) )
            return h
        raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)

    def calcFieldsDeriv(self, sol, freq, fieldType, v, adjoint=False):
        j = sol
        if fieldType == 'j':
            return None
        elif fieldType == 'h':
            MeMuI = self.MeMuI
            C = self.mesh.edgeCurl
            sig = self.curModel.transform
            sigi = 1/sig
            dsig_dm = self.curModel.transformDeriv
            dsigi_dsig = -Utils.sdiag(sigi)**2
            dMf_dsigi = self.mesh.getFaceInnerProductDeriv(sigi)(j)
            sigi = self.MfSigmai
            if not adjoint:
                return -(1./(1j*omega(freq))) * MeMuI * ( C.T * ( dMf_dsigi * ( dsigi_dsig * ( dsig_dm * v ) ) ) )
            else:
                return -(1./(1j*omega(freq))) * dsig_dm.T * ( dsigi_dsig.T * ( dMf_dsigi.T * ( C * ( MeMuI.T * v ) ) ) )
        raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)




# Solving for h! - using primary- secondary approach
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

    solType = 'h'
    storeTheseFields = ['j','h']

    def __init__(self, model, **kwargs):
        BaseFDEMProblem.__init__(self, model, **kwargs)

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
            :rtype: numpy.ndarray (nE, nTx)
            :return: RHS
        """
        b_0 = getSource(self,freq)
        return -1j*omega(freq)*b_0

    def calcFields(self, sol, freq, fieldType, adjoint=False):
        h = sol
        if fieldType == 'j':
            C = self.mesh.edgeCurl
            if adjoint:
                return C.T*h
            return C*h
        elif fieldType == 'h':
            return h
        raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)

    def calcFieldsDeriv(self, sol, freq, fieldType, v, adjoint=False):
        return None
