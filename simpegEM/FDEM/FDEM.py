from SimPEG import Survey, Problem, Utils, np, sp, Solver as SimpegSolver
from scipy.constants import mu_0
from SurveyFDEM import SurveyFDEM, FieldsFDEM
from simpegEM import Sources
from simpegEM.Base import BaseEMProblem

def omega(freq):
    """Change frequency to angular frequency, omega"""
    return 2.*np.pi*freq

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
        Txs = self.survey.getTransmitters(freq)
        rhs = range(len(Txs))
        for i, tx in enumerate(Txs):
            if tx.txType == 'VMD':
                src = Sources.MagneticDipoleVectorPotential
                SRCx = src(tx.loc, self.mesh.gridEx, 'x')
                SRCy = src(tx.loc, self.mesh.gridEy, 'y')
                SRCz = src(tx.loc, self.mesh.gridEz, 'z')

            elif tx.txType == 'CircularLoop':
                src = Sources.MagneticLoopVectorPotential                
                SRCx = src(tx.loc, self.mesh.gridEx, 'x', tx.radius)
                SRCy = src(tx.loc, self.mesh.gridEy, 'y', tx.radius)
                SRCz = src(tx.loc, self.mesh.gridEz, 'z', tx.radius)
            else:
                raise NotImplemented('%s txType is not implemented' % tx.txType)
            rhs[i] = np.concatenate((SRCx, SRCy, SRCz))

        a = np.concatenate(rhs).reshape((self.mesh.nE, len(Txs)), order='F')
        mui = self.MfMui
        C = self.mesh.edgeCurl

        j_s = C.T*mui*C*a
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

        return mui*C*sigI*C.T*mui + 1j*omega(freq)*mui

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
            return dsig_dm.T * ( dMe_dsig.T * ( dMeSigmaI_dI.T * ( C.T * ( mui.T * v ) ) ) )

        return mui * ( C * ( dMeSigmaI_dI * ( dMe_dsig * ( dsig_dm * v ) ) ) )

    def getRHS(self, freq):
        """
            :param float freq: Frequency
            :rtype: numpy.ndarray (nE, nTx)
            :return: RHS
        """
        Txs = self.survey.getTransmitters(freq)
        rhs = range(len(Txs))
        for i, tx in enumerate(Txs):

            if self.mesh._meshType is 'CYL':
                if self.mesh.isSymmetric:
                    if tx.txType == 'VMD':                    
                        SRC = Sources.MagneticDipoleVectorPotential(tx.loc, self.mesh.gridEy, 'y')
                    elif tx.txType =='CircularLoop':
                        SRC = Sources.MagneticLoopVectorPotential(tx.loc, self.mesh.gridEy, 'y', tx.radius)
                    else:
                        raise NotImplementedError('Only VMD and CircularLoop')
                else:
                    raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')
    
            elif self.mesh._meshType is 'TENSOR':

                if tx.txType == 'VMD':
                    src = Sources.MagneticDipoleVectorPotential
                    SRCx = src(tx.loc, self.mesh.gridEx, 'x')
                    SRCy = src(tx.loc, self.mesh.gridEy, 'y')
                    SRCz = src(tx.loc, self.mesh.gridEz, 'z')

                elif tx.txType == 'VMD_B':
                    src = Sources.MagneticDipoleFields
                    SRCx = src(tx.loc, self.mesh.gridFx, 'x')
                    SRCy = src(tx.loc, self.mesh.gridFy, 'y')
                    SRCz = src(tx.loc, self.mesh.gridFz, 'z')

                elif tx.txType == 'CircularLoop':
                    src = Sources.MagneticLoopVectorPotential                
                    SRCx = src(tx.loc, self.mesh.gridEx, 'x', tx.radius)
                    SRCy = src(tx.loc, self.mesh.gridEy, 'y', tx.radius)
                    SRCz = src(tx.loc, self.mesh.gridEz, 'z', tx.radius)
                else:

                    raise NotImplemented('%s txType is not implemented' % tx.txType)
                SRC = np.concatenate((SRCx, SRCy, SRCz))                                    

            else:
                raise Exception('Unknown mesh for VMD')           
            
            rhs[i] = SRC
            
        mui = self.MfMui            
        if tx.txType == 'VMD_B':
            b_0 = np.concatenate(rhs).reshape((self.mesh.nF, len(Txs)), order='F')
        else:            
            a = np.concatenate(rhs).reshape((self.mesh.nE, len(Txs)), order='F')
            C = self.mesh.edgeCurl
            b_0 = C*a

        return -1j*omega(freq)*mui*b_0

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

        return C * MeMuI * C.T * MfSigi + iomega


    def getADeriv(self, freq, u, v, adjoint=False):
        """
            In this case, we assume that electrical conductivity, \(\\sigma\) is the physical property of interest (i.e. \(\sigma\) = model.transform). Then we want
            .. math:: 
                \\frac{\mathbf{A(\\sigma)} \mathbf{v}}{d \\mathbf{m}} &= \\mathbf{C}  \\mathbf{M^e_{mu^{-1}}} \\mathbf{C^T} \\frac{d \\mathbf{M^f_{\\sigma^{-1}}}}{d \\mathbf{m}}
                &= \\mathbf{C}  \\mathbf{M^e_{mu}^{-1}} \\mathbf{C^T} \\frac{d \\mathbf{M^f_{\\sigma^{-1}}}}{d \\mathbf{\\sigma^{-1}}} \\frac{d \\mathbf{\\sigma^{-1}}}{d \\mathbf{\\sigma}} \\frac{d \\mathbf{\\sigma}}{d \\mathbf{m}}
        """

        MeMuI = self.MeMuI
        C = self.mesh.edgeCurl
        sig = self.curModel.transform
        sigi = 1/sig
        dsig_dm = self.curModel.transformDeriv
        dsigi_dsig = -Utils.sdiag(sigi)**2
        dMf_dsigi = self.mesh.getFaceInnerProductDeriv(sigi)(u)

        if adjoint:
            return dsig_dm.T * ( dsigi_dsig.T *( dMf_dsigi.T * ( C * ( MeMuI.T * ( C.T * v ) ) ) ) )

        return C * ( MeMuI * ( C.T * ( dMf_dsigi * ( dsigi_dsig * ( dsig_dm * v ) ) ) ) ) 


    def getRHS(self, freq):
        """
            :param float freq: Frequency
            :rtype: numpy.ndarray (nE, nTx)
            :return: RHS
        """
        Txs = self.survey.getTransmitters(freq)
        rhs = range(len(Txs))
        for i, tx in enumerate(Txs):
            if tx.txType == 'VMD':
                src = Sources.MagneticDipoleVectorPotential
                SRCx = src(tx.loc, self.mesh.gridFx, 'x')
                SRCy = src(tx.loc, self.mesh.gridFy, 'y')
                SRCz = src(tx.loc, self.mesh.gridFz, 'z')

            elif tx.txType == 'CircularLoop':
                src = Sources.MagneticLoopVectorPotential                
                SRCx = src(tx.loc, self.mesh.gridFx, 'x', tx.radius)
                SRCy = src(tx.loc, self.mesh.gridFy, 'y', tx.radius)
                SRCz = src(tx.loc, self.mesh.gridFz, 'z', tx.radius)
            else:
                raise NotImplemented('%s txType is not implemented' % tx.txType)
            rhs[i] = np.concatenate((SRCx, SRCy, SRCz))

        a = np.concatenate(rhs).reshape((self.mesh.nF, len(Txs)), order='F')
        a = Utils.mkvc(a)
        MeMuI = self.MeMuI
        C = self.mesh.edgeCurl

        j_s = C*MeMuI*C.T*a
        return -1j*omega(freq)*j_s

    def calcFields(self, sol, freq, fieldType, adjoint=False):
        j = sol
        if fieldType == 'j':
            return j
        elif fieldType == 'h':
            mui = self.MeMuI
            C = self.mesh.edgeCurl
            MfSigi = self.MfSigmai
            if not adjoint:
                h = -(1./(1j*omega(freq))) * mui * ( C.T * ( MfSigi * j ) )
            else:
                h = -(1./(1j*omega(freq))) * MfSigi * ( C * ( mui.T * j ) ) 
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




# Solving for h! - NOTE: WE ARE GOING TO NEED dRHS / dm !
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


    def getjs(self,freq):
        """
            :param float freq: Frequency
            :rtype: numpy.ndarray (nE, nTx)
            :return: j_s
        """
        Txs = self.survey.getTransmitters(freq)
        rhs = range(len(Txs))
        for i, tx in enumerate(Txs):
            if tx.txType == 'VMD':
                src = Sources.MagneticDipoleVectorPotential
                SRCx = src(tx.loc, self.mesh.gridFx, 'x')
                SRCy = src(tx.loc, self.mesh.gridFy, 'y')
                SRCz = src(tx.loc, self.mesh.gridFz, 'z')

            elif tx.txType == 'CircularLoop':
                src = Sources.MagneticLoopVectorPotential                
                SRCx = src(tx.loc, self.mesh.gridFx, 'x', tx.radius)
                SRCy = src(tx.loc, self.mesh.gridFy, 'y', tx.radius)
                SRCz = src(tx.loc, self.mesh.gridFz, 'z', tx.radius)
            else:
                raise NotImplemented('%s txType is not implemented' % tx.txType)
            rhs[i] = np.concatenate((SRCx, SRCy, SRCz))

        a = np.concatenate(rhs).reshape((self.mesh.nF, len(Txs)), order='F')
        a = Utils.mkvc(a)

        MeMuI = self.MeMuI
        C = self.mesh.edgeCurl

        return C*MeMuI*C.T*a

    def getRHS(self, freq):
        """
            :param float freq: Frequency
            :rtype: numpy.ndarray (nE, nTx)
            :return: RHS
        """
        MfSigi = self.MfSigmai
        C = self.mesh.edgeCurl
        j_s = self.getjs(freq)
        return C.T*MfSigi*j_s

    def getRHSDeriv(self, freq, v, adjoint=False):
        """
            :param float freq: Frequency
            :rtype: numpy.ndarray (nE, nTx)
            :return: RHSDeriv
        """
        C = self.mesh.edgeCurl
        sig = self.curModel.transform
        sigi = 1/sig
        j_s = self.getjs(freq)
        dMf_dsigi = self.mesh.getFaceInnerProductDeriv(sigi)(j_s)
        dsig_dm = self.curModel.transformDeriv
        dsigi_dsig = -Utils.sdiag(sigi)**2 # only works for diagonal matrices 

        if adjoint:
            return dsig_dm.T * dsigi_dsig.T * dMf_dsigi.T * C * v
        return C.T * dMf_dsigi * dsigi_dsig * dsig_dm * v
        
    def calcFields(self, sol, freq, fieldType, adjoint=False):
        h = sol
        if fieldType == 'j':
            NotImplementedError('fieldType "%s" is not implemented.' % fieldType)
            # C = self.mesh.edgeCurl
            # j_s = self.getjs(freq)
            # if adjoint: 
            #     MeMu = self.MeMu
            #     MfSigi = self.MfSigmai
            #     MfSigmaiinv = self.Solver(MfSigi.T, **self.solverOpts)
            #     Cinv = self.Solver(C, **self.solverOpts)
            #     return  -1j * omega(freq) * MeMu.T * (MfSigmaiinv * (CTinv * h))
            # return C*h - j_s # -iomega(freq) inv(MfSigmai) inv(C.T) MeMu
        elif fieldType == 'h':
            return h
        raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)

    def calcFieldsDeriv(self, sol, freq, fieldType, v, adjoint=False):
        h = sol
        A = self.getA(freq)

        if fieldType == 'j':
            # C = self.mesh.edgeCurl
            # MeMu = self.MeMu
            # MfSigi = self.MfSigmai
            # MfSigmaiinv = self.Solver(MfSigi, **self.solverOpts)
            # CTinv = self.Solver(C.T, **self.solverOpts)
            # hDeriv = self.calcFieldsDeriv(h,freq,'h',v,adjoint=False)

            # pt1 = -1j * omega(freq) * (MfSigmaiinv * (CTinv * (MeMu * hDeriv)))

            # sig = self.curModel.transform
            # sigi = 1/sig
            # dsig_dm = self.curModel.transformDeriv
            # dsigi_dsig = -Utils.sdiag(sigi)**2

            # v1 = MfSigmaiinv *(CTinv * (MeMu * h))

            # dMf_dsigi = self.mesh.getFaceInnerProductDeriv(sigi)(v1)

            # pt2 = 1j * omega(freq) * (MfSigmaiinv * (dMf_dsigi * (dsigi_dsig * (dsig_dm * v))))

            # return pt1+pt2


            # if adjoint:
            #     MfSigmaiTinv = self.Solver(MfSigi.T, **self.solverOpts)
            #     Cinv = self.Solver(C, **self.solverOpts)
            #     MeMu.T * (Cinv * (MfSigmaiTinv * h))
            #     v1 = MeMu.T *( Cinv *(MfSigmaiTinv * v))
            #     pt1 = -1j * omega(freq) * self.calcFieldsDeriv(h,freq,'h',v,adjoint=True)

            #     pt2 =  
            raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)

        elif fieldType == 'h':
            if adjoint:
                ATinv = self.Solver(A.T, **self.solverOpts)
                ATinvv = ATinv*v
                return self.getRHSDeriv(freq,ATinvv,adjoint=True) 
            dRHSh = self.getRHSDeriv(freq,v,adjoint)
            Ainv = self.Solver(A, **self.solverOpts)
            return Ainv*dRHSh
        raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)