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
            sigma = self.currentTransformedModel
            self._MeSigma = self.mesh.getEdgeInnerProduct(sigma)
        return self._MeSigma

    @property
    def MeSigmaI(self):
        # TODO: this will not work if tensor conductivity
        if getattr(self, '_MeSigmaI', None) is None:
            self._MeSigmaI = Utils.sdiag(1/self.MeSigma.diagonal())
        return self._MeSigmaI

    currentTransformedModel = Utils.dependentProperty('_currentTransformedModel', None, ['_MeSigma', '_MeSigmaI'], 'Sets the current model, and removes dependent mass matrices.')

    def fields(self, m):
        self.currentTransformedModel = self.model.transform(m)
        F = self.forward(m, self.getRHS, self.calcFields)
        return F

    def forward(self, m, RHS, CalcFields):

        F = FieldsFDEM(self.mesh, self.survey)

        for freq in self.survey.freqs:
            A = self.getA(freq)
            rhs = RHS(freq)
            solver = self.Solver(A, **self.solverOpts)
            sol = solver.solve(rhs)
            F[freq] = CalcFields(sol, freq)

        return F

class ProblemFDEM_e(BaseProblemFDEM):
    """
        Solving for e!
    """

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

    def calcFields(self, sol, freq):
        e = sol

        fDict = {}

        if 'e' in self.storeTheseFields:
            fDict['e'] = e

        if 'b' in self.storeTheseFields:
            b = -1./(1j*omega(freq))*self.mesh.edgeCurl*e
            fDict['b'] = b

        return fDict

    def Jvec(self, m, v, u=None):
        if u is None:
           u = self.fields(m)

        sig = self.model.transform(m)
        self.currentTransformedModel = sig

        Jv = self.dataPair(self.survey)
        dsig_dm = self.model.transformDeriv(m)

        for i, freq in enumerate(self.survey.freqs):
            e = u[freq, 'e']
            A = self.getA(freq)
            solver = self.Solver(A, **self.solverOpts)

            for txi, tx in enumerate(self.survey.getTransmitters(freq)):
                dMe_dsig = self.mesh.getEdgeInnerProductDeriv(sig, v=e[:,txi])

                P = tx.projectFieldsDeriv(self.mesh, u)
                b = 1j*omega(freq) * ( dMe_dsig * ( dsig_dm * v ) )
                Ainvb = solver.solve(b)
                Jv[tx] = -P*Ainvb

        return Utils.mkvc(Jv)


    def Jtvec(self, m, v, u=None):
        if u is None:
            u = self.fields(m)

        sig = self.model.transform(m)
        self.currentTransformedModel = sig

        # Ensure v is a data object.
        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        Jtv = np.zeros(self.model.nP, dtype=complex)

        dsig_dm = self.model.transformDeriv(m)

        for i, freq in enumerate(self.survey.freqs):
            e = u[freq, 'e']
            AT = self.getA(freq).T
            solver = self.Solver(AT, **self.solverOpts)

            for txi, tx in enumerate(self.survey.getTransmitters(freq)):
                dMe_dsig = self.mesh.getEdgeInnerProductDeriv(sig, v=e[:,txi])

                P  = tx.projectFieldsDeriv(self.mesh, u)
                w = solver.solve(P.T * v[tx])
                Jtv += - 1j*omega(freq) * ( dsig_dm.T * ( dMe_dsig.T * w ) )

        return Jtv

class ProblemFDEM_b(BaseProblemFDEM):
    """
        Solving for b!
    """

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

    def calcFields(self, sol, freq):
        b = sol

        fDict = {}

        if 'b' in self.storeTheseFields:
            fDict['b'] = b

        if 'e' in self.storeTheseFields:
            e = self.MeSigmaI*self.mesh.edgeCurl.T*self.MfMui*b
            fDict['e'] = e

        return fDict

    def Jvec(self, m, v, u=None):
        if u is None:
           u = self.fields(m)

        raise NotImplemented('')

        # sig = self.model.transform(m)
        # self.currentTransformedModel = sig

        # Jv = self.dataPair(self.survey)
        # dsig_dm = self.model.transformDeriv(m)

        # for i, freq in enumerate(self.survey.freqs):
        #     e = u[freq, 'e']
        #     A = self.getA(freq)
        #     solver = self.Solver(A, **self.solverOpts)

        #     for txi, tx in enumerate(self.survey.getTransmitters(freq)):
        #         dMe_dsig = self.mesh.getEdgeInnerProductDeriv(sig, v=e[:,txi])

        #         P = tx.projectFieldsDeriv(self.mesh, u)
        #         b = 1j*omega(freq) * ( dMe_dsig * ( dsig_dm * v ) )
        #         Ainvb = solver.solve(b)
        #         Jv[tx] = -P*Ainvb

        # return Utils.mkvc(Jv)


    def Jtvec(self, m, v, u=None):
        if u is None:
            u = self.fields(m)

        # Ensure v is a data object.
        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        raise NotImplemented('')

        # sig = self.model.transform(m)
        # self.currentTransformedModel = sig

        # Jtv = np.zeros(self.model.nP, dtype=complex)

        # dsig_dm = self.model.transformDeriv(m)

        # for i, freq in enumerate(self.survey.freqs):
        #     e = u[freq, 'e']
        #     AT = self.getA(freq).T
        #     solver = self.Solver(AT, **self.solverOpts)

        #     for txi, tx in enumerate(self.survey.getTransmitters(freq)):
        #         dMe_dsig = self.mesh.getEdgeInnerProductDeriv(sig, v=e[:,txi])

        #         P  = tx.projectFieldsDeriv(self.mesh, u)
        #         w = solver.solve(P.T * v[tx])
        #         Jtv += - 1j*omega(freq) * ( dsig_dm.T * ( dMe_dsig.T * w ) )

        # return Jtv
