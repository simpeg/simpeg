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

    solType = 'b'
    storeTheseFields = 'e'

    surveyPair = SurveyFDEM
    dataPair = DataFDEM

    Solver = SimpegSolver
    solverOpts = {'doDirect':True, 'options':{'factorize':False, 'backend':'scipy'}}

    ####################################################
    # Mass Matrices
    ####################################################

    @property
    def MfMui(self): return self._MfMui

    @property
    def Me(self): return self._Me

    @property
    def MeSigma(self): return self._MeSigma

    @property
    def MeSigmaI(self): return self._MeSigmaI

    def makeMassMatrices(self, m):
        #TODO: hardcoded to sigma as the model
        sigma = self.model.transform(m)
        self._Me = self.mesh.getEdgeInnerProduct()
        self._MeSigma = self.mesh.getEdgeInnerProduct(sigma)
        # TODO: this will not work if tensor conductivity
        self._MeSigmaI = Utils.sdiag(1/self.MeSigma.diagonal())
        #TODO: assuming constant mu
        self._MfMui = self.mesh.getFaceInnerProduct(1/mu_0)


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
        return self.mesh.edgeCurl.T*self.MfMui*self.mesh.edgeCurl + 1j*omega(freq)*self.MeSigma

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
        C = self.mesh.edgeCurl
        j_s = C.T*self.MfMui*C*a
        #TODO: self.Me* ??
        return -1j*omega(freq)*j_s


    def fields(self, m, useThisRhs=None):
        RHS = useThisRhs or self.getRHS

        self.makeMassMatrices(m)

        F = FieldsFDEM(self.mesh, self.survey)

        for freq in self.survey.freqs:
            A = self.getA(freq)
            rhs = self.getRHS(freq)
            solver = self.Solver(A, **self.solverOpts)
            e = solver.solve(rhs)

            print np.linalg.norm(A*Utils.mkvc(e) - Utils.mkvc(rhs)) / np.linalg.norm(Utils.mkvc(rhs))

            F[freq, 'e'] = e
            b = -1./(1j*omega(freq))*self.mesh.edgeCurl*e
            F[freq, 'b'] = b

        return F


    def Jvec(self, m, v, u=None):
        if u is None:
           u = self.fields(m)

        Jv = self.dataPair(self.survey)
        sig = self.model.transform(m)
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

        # Ensure v is a data object.
        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        Jtv = np.zeros(self.model.nP, dtype=complex)

        sig = self.model.transform(m)
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
        return self.MfMui*self.mesh.edgeCurl*self.MeSigmaI*self.mesh.edgeCurl.T*self.MfMui + 1j*omega(freq)*self.MfMui

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
        C = self.mesh.edgeCurl
        b_0 = C*a
        return -1j*omega(freq)*self.MfMui*b_0


    def fields(self, m, useThisRhs=None):
        RHS = useThisRhs or self.getRHS

        self.makeMassMatrices(m)

        F = FieldsFDEM(self.mesh, self.survey)

        for freq in self.survey.freqs:
            A = self.getA(freq)
            rhs = self.getRHS(freq)
            solver = self.Solver(A, **self.solverOpts)
            #Note that we are solving for b_s
            b = solver.solve(rhs)

            print np.linalg.norm(A*Utils.mkvc(b) - Utils.mkvc(rhs)) / np.linalg.norm(Utils.mkvc(rhs))

            F[freq, 'b'] = b
            e = self.MeSigmaI*self.mesh.edgeCurl.T*self.MfMui*b
            F[freq, 'e'] = e

        return F


    def Jvec(self, m, v, u=None):
        if u is None:
           u = self.fields(m)

        raise NotImplemented('')

        # Jv = self.dataPair(self.survey)
        # sig = self.model.transform(m)
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
        # Jtv = np.zeros(self.model.nP, dtype=complex)

        # sig = self.model.transform(m)
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
