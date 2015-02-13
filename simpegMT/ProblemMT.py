from SimPEG import Survey, Problem, Utils, np, sp, Solver as SimpegSolver
from scipy.constants import mu_0
from SurveyMT import SurveyMT, FieldsMT


def omega(freq):
    """Change frequency to angular frequency, omega"""
    return 2.*np.pi*freq


class MTProblem(Problem.BaseProblem):

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

    solType = 'e'
    storeTheseFields = ['e', 'b']

    surveyPair = SurveyMT
    dataPair = Survey.Data

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

    # TODO:
    # MeSigmaBG

    @property
    def MeSigmaI(self):
        #TODO: hardcoded to sigma as the model
        if getattr(self, '_MeSigmaI', None) is None:
            sigma = self.curTModel
            self._MeSigmaI = self.mesh.getEdgeInnerProduct(sigma, invMat=True)
        return self._MeSigmaI

    curModel = Utils.dependentProperty('_curModel', None, ['_MeSigma', '_MeSigmaI', '_curTModel', '_curTModelDeriv'], 'Sets the current model, and removes dependent mass matrices.')

    @property
    def curTModel(self):
        if getattr(self, '_curTModel', None) is None:
            self._curTModel = self.mapping.transform(self.curModel)
        return self._curTModel

    @property
    def curTModelDeriv(self):
        if getattr(self, '_curTModelDeriv', None) is None:
            self._curTModelDeriv = self.mapping.transformDeriv(self.curModel)
        return self._curTModelDeriv

    def fields(self, m):
        self.curModel = m
        RHS, CalcFields = self.getRHS, self.calcFields

        F = FieldsMT(self.mesh, self.survey)

        for freq in self.survey.freqs:
            A = self.getA(freq)
            rhs = RHS(freq)
            Ainv = self.Solver(A, **self.solverOpts)
            e = Ainv * rhs # is this e?

            Src = self.survey.getSources(freq)
            # Stroe the fields
            F[Src, 'e'] = e
            F[Src, 'b'] = self.mesh.edgeCurl * e 

        return F


    def getA(self, freq):
        """
            Function to get the A matrix.

            :param float freq: Frequency
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """
        mui = self.MfMui
        sig = self.MeSigma
        C = self.mesh.edgeCurl

        return C.T*mui*C + 1j*omega(freq)*sig
    def getAbg(self, freq):
        """
            Function to get the A matrix for the background model.

            :param float freq: Frequency
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """
        mui = self.MfMui
        sigBG = self.MeSigmaBG
        C = self.mesh.edgeCurl

        return C.T*mui*C + 1j*omega(freq)*sigBG

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
            :rtype: numpy.ndarray (nE, 2)
            :return: one RHS for both polarizations
        """
        raise NotImplementedError()

        """
        Put this in MT.Sources.EldadsSource

        """

        Txs = self.survey.getTransmitters(freq)

        # assert that only one Tx/src?
        # Create the two polarizations at this freq and return np array (nE,2).

        # solve analytic.... get p1 p2

        # Abg * [p1,p2] = rhs

        rhs = range(len(Txs))
        for i, tx in enumerate(Txs):
            if tx.txType == 'VMD': # EH source.
                src = Sources.MagneticDipoleVectorPotential # this is where you would put multiple types of boundary conditions.
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

    ##################################################################
    # Inversion stuff
    ##################################################################
    # Not really used now....


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
            solver = self.Solver(AT, **self.solverOpts)

            for tx in self.survey.getTransmitters(freq):
                u_tx = u[tx, self.solType]

                for rx in tx.rxList:
                    PTv = rx.projectFieldsDeriv(tx, self.mesh, u, v[tx, rx], adjoint=True)
                    fPTv = self.calcFields(PTv, freq, rx.projField, adjoint=True)

                    w = solver.solve( fPTv )
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
