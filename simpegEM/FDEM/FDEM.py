from SimPEG import Problem, Utils, np, sp, Solver as SimpegSolver
from scipy.constants import mu_0
from SurveyFDEM import SurveyFDEM, DataFDEM, FieldsFDEM
from simpegEM.Utils import Sources

def omega(freq):
    """Change frequency to angular frequency, omega"""
    return 2.*np.pi*freq

class BaseProblemFDEM(Problem.BaseProblem):
    """
        Frequency-Domain EM problem - E-formulation


        .. math::

            \dcurl E + i \omega B = 0 \\\\
            \dcurl^\\top \MfMui B - \MeSig E = \Me \j_s
    """
    def __init__(self, model, **kwargs):
        Problem.BaseProblem.__init__(self, model, **kwargs)

    solType = None
    storeTheseFields = ['e', 'b']

    surveyPair = SurveyFDEM
    dataPair = DataFDEM

    Solver = SimpegSolver
    solverOpts = {}

    ####################################################
    # Mass Matrices
    ####################################################

    @property
    def MfMui(self):
        #TODO: assuming constant mu
        if getattr(self, '_MfMui', None) is None:
            self._MfMui = self.mesh.getFaceInnerProduct(1/mu_0)
        return self._MfMui

    @property
    def Me(self):
        if getattr(self, '_Me', None) is None:
            self._Me = self.mesh.getEdgeInnerProduct()
        return self._Me

    @property
    def MeSigma(self):
        #TODO: hardcoded to sigma as the model
        if getattr(self, '_MeSigma', None) is None:
            sigma = self.curTModel
            self._MeSigma = self.mesh.getEdgeInnerProduct(sigma)
        return self._MeSigma

    @property
    def MeSigmaI(self):
        # TODO: this will not work if tensor conductivity
        if getattr(self, '_MeSigmaI', None) is None:
            self._MeSigmaI = Utils.sdiag(1/self.MeSigma.diagonal())
        return self._MeSigmaI

    curModel = Utils.dependentProperty('_curModel', None, ['_MeSigma', '_MeSigmaI', '_curTModel', '_curTModelDeriv'], 'Sets the current model, and removes dependent mass matrices.')

    @property
    def curTModel(self):
        if getattr(self, '_curTModel', None) is None:
            self._curTModel = self.model.transform(self.curModel)
        return self._curTModel

    @property
    def curTModelDeriv(self):
        if getattr(self, '_curTModelDeriv', None) is None:
            self._curTModelDeriv = self.model.transformDeriv(self.curModel)
        return self._curTModelDeriv

    def fields(self, m):
        self.curModel = m
        F = self.forward(m, self.getRHS, self.calcFields)
        return F

    def forward(self, m, RHS, CalcFields):

        F = FieldsFDEM(self.mesh, self.survey)

        for freq in self.survey.freqs:
            A = self.getA(freq)
            rhs = RHS(freq)
            solver = self.Solver(A, **self.solverOpts)
            sol = solver.solve(rhs)
            for fieldType in self.storeTheseFields:
                F[freq, fieldType] = CalcFields(sol, freq, fieldType)

        return F

    def Jvec(self, m, v, u=None):
        if u is None:
           u = self.fields(m)

        self.curModel = m

        Jv = self.dataPair(self.survey)

        for freq in self.survey.freqs:
            A = self.getA(freq)
            solver = self.Solver(A, **self.solverOpts)

            for tx in self.survey.getTransmitters(freq):
                u_tx = u[tx, self.solType]
                w = self.getADeriv(freq, u_tx, v)
                Ainvw = solver.solve(w)
                fAinvw = self._calcFieldsList(Ainvw, freq, tx.rxList.fieldTypes)
                P = lambda v: tx.projectFieldsDeriv(self.mesh, u, v)

                df_dm = self._calcFieldsDerivList(u_tx, freq, tx.rxList.fieldTypes, v)
                #TODO: this is now a list?
                if df_dm is None:
                    Jv[tx] = - P(fAinvw)
                else:
                    Jv[tx] = - P(fAinvw) + P(df_dm)

        return Utils.mkvc(Jv)

    def Jtvec(self, m, v, u=None):
        if u is None:
            u = self.fields(m)

        self.curModel = m

        # Ensure v is a data object.
        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        Jtv = np.zeros(self.model.nP, dtype=complex)

        for freq in self.survey.freqs:
            AT = self.getA(freq).T
            solver = self.Solver(AT, **self.solverOpts)

            for tx in self.survey.getTransmitters(freq):
                u_tx = u[tx, self.solType]

                PTv = tx.projectFieldsDeriv(self.mesh, u, v[tx], adjoint=True)
                fPTv = self._calcFieldsList(PTv, freq, tx.rxList.fieldTypes, adjoint=True)

                w = solver.solve( fPTv )
                Jtv_tx = self.getADeriv(freq, u_tx, w, adjoint=True)

                df_dm = self._calcFieldsDerivList(u_tx, freq, tx.rxList.fieldTypes, PTv, adjoint=True)

                if df_dm is None:
                    Jtv += - Jtv_tx
                else:
                    Jtv += - Jtv_tx + df_dm

        return Jtv

    def _calcFieldsList(self, sol, freq, fieldTypes, adjoint=False):
        fVecs = range(len(fieldTypes))
        for ii, fieldType in enumerate(fieldTypes):
            fVecs[ii] = self.calcFields(sol, freq, fieldType, adjoint=adjoint)
        return np.concatenate(fVecs)

    def _calcFieldsDerivList(self, sol, freq, fieldTypes, v, adjoint=False):
        fVecs = range(len(fieldTypes))
        V = v.reshape((-1, len(fieldTypes)), order='F')
        for ii, fieldType in enumerate(fieldTypes):
            fVecs[ii] = self.calcFieldsDeriv(sol, freq, fieldType, V[:,ii], adjoint=adjoint)
        return np.concatenate(fVecs)

class ProblemFDEM_e(BaseProblemFDEM):
    """
        Solving for e!
    """
    solType = 'e'

    def __init__(self, model, **kwargs):
        BaseProblemFDEM.__init__(self, model, **kwargs)

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
        sig = self.curTModel
        dsig_dm = self.curTModelDeriv
        dMe_dsig = self.mesh.getEdgeInnerProductDeriv(sig, v=u)

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
            else:
                raise NotImplemented('%s txType is not implemented' % tx.txType)
            SRCx = src(tx.loc, self.mesh.gridEx, 'x')
            SRCy = src(tx.loc, self.mesh.gridEy, 'y')
            SRCz = src(tx.loc, self.mesh.gridEz, 'z')
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
                b = -1./(1j*omega(freq)) * ( self.mesh.edgeCurl * e )
            else:
                b = -1./(1j*omega(freq)) * ( self.mesh.edgeCurl.T * e )
            return b
        raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)

    def calcFieldsDeriv(self, sol, freq, fieldType, v, adjoint=False):
        e = sol
        if fieldType == 'e':
            return None
        elif fieldType == 'b':
            return None
        raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)


class ProblemFDEM_b(BaseProblemFDEM):
    """
        Solving for b!
    """
    solType = 'b'

    def __init__(self, model, **kwargs):
        BaseProblemFDEM.__init__(self, model, **kwargs)

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
        sig = self.curTModel
        dsig_dm = self.curTModelDeriv
        #TODO: This only works if diagonal (no tensors)...
        dMeSigmaI_dI = - self.MeSigmaI**2

        vec = (C.T*(mui*u))
        dMe_dsig = self.mesh.getEdgeInnerProductDeriv(sig, v=vec)

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
            if tx.txType == 'VMD':
                src = Sources.MagneticDipoleVectorPotential
            else:
                raise NotImplemented('%s txType is not implemented' % tx.txType)
            SRCx = src(tx.loc, self.mesh.gridEx, 'x')
            SRCy = src(tx.loc, self.mesh.gridEy, 'y')
            SRCz = src(tx.loc, self.mesh.gridEz, 'z')
            rhs[i] = np.concatenate((SRCx, SRCy, SRCz))

        a = np.concatenate(rhs).reshape((self.mesh.nE, len(Txs)), order='F')
        mui = self.MfMui
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
            sig = self.curTModel
            dsig_dm = self.curTModelDeriv

            C = self.mesh.edgeCurl
            mui = self.MfMui

            #TODO: This only works if diagonal (no tensors)...
            dMeSigmaI_dI = - self.MeSigmaI**2

            vec = C.T * ( mui * b )
            dMe_dsig = self.mesh.getEdgeInnerProductDeriv(sig, v=vec)
            if not adjoint:
                return dMeSigmaI_dI * ( dMe_dsig * ( dsig_dm * v ) )
            else:
                return dsig_dm.T * ( dMe_dsig.T * ( dMeSigmaI_dI.T * v ) )
        elif fieldType == 'b':
            return None
        raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)

