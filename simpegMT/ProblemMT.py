from SimPEG import Survey, Problem, Utils, Models, np, sp, SolverLU as SimpegSolver
from scipy.constants import mu_0
from SurveyMT import SurveyMT, FieldsMT
import multiprocessing, sys, time

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
            self._curTModel = self.mapping*self.curModel
        return self._curTModel

    @property
    def curTModelDeriv(self):
        if getattr(self, '_curTModelDeriv', None) is None:
            self._curTModelDeriv = self.mapping*self.curModel
        return self._curTModelDeriv


    # Background model
    @property
    def backModel(self):
        """
            Sets the model, and removes dependent mass matrices.
        """
        return getattr(self, '_backModel', None)

    @backModel.setter
    def backModel(self, value):
        if value is self.backModel:
            return # it is the same!
        self._backModel = Models.Model(value, self.mapping)
        for prop in self.deleteTheseOnModelUpdate:
            if hasattr(self, prop):
                delattr(self, prop)

    @property
    def MeSigmaBG(self):
        #TODO: hardcoded to sigma as the model
        if getattr(self, '_MeSigmaBG', None) is None:
            sigmaBG = self.backModel
            self._MeSigmaBG = self.mesh.getEdgeInnerProduct(sigmaBG)
        return self._MeSigmaBG

    def fields(self, m, m_back,nrProc=None):
        '''
        Function to calculate all the fields for the model m.

        :param np.ndarray (nC,) m: Conductivity model
        :param np.ndarray (nC,) m_back: Background conductivity model
        '''
        self.curModel = m
        self.backModel = m_back
        # RHS, CalcFields = self.getRHS(freq,m_back), self.calcFields

        F = FieldsMT(self.mesh, self.survey)
        startTime = time.time()
        def solveAtFreq(self,F,freq):
            print 'Starting work for {:.3e}'.format(freq)
            sys.stdout.flush()
            A = self.getA(freq)
            rhs = self.getRHS(freq,m_back)
            Ainv = self.Solver(A, **self.solverOpts)
            e = Ainv * rhs 

            # Store the fields
            Src = self.survey.getSources(freq)
            # Store the fieldss
            F[Src, 'e_px'] = e[:,0]
            F[Src, 'e_py'] = e[:,1]
            # Note curl e = -iwb so b = -curl/iw
            b = -( self.mesh.edgeCurl * e )/( 1j*omega(freq) )
            F[Src, 'b_px'] = b[:,0]
            F[Src, 'b_py'] = b[:,1]
            print 'Ran for {:f} seconds'.format(time.time()-startTime)
            sys.stdout.flush()
            return F
        #NOTE: add print status statements.
        if nrProc is None:
            for freq in self.survey.freqs:
                F = solveAtFreq(self,F,freq)
        else:
            pool = multiprocessing.Pool(processes=nrProc)
            pool.map(solveAtFreq,self.survey.freqs)
            pool.close()
            pool.join()

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

        return C.T*mui*C - 1j*omega(freq)*sig

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

        return C.T*mui*C - 1j*omega(freq)*sigBG

    def getADeriv(self, freq, u, v, adjoint=False):
        sig = self.curTModel
        dsig_dm = self.curTModelDeriv
        dMe_dsig = self.mesh.getEdgeInnerProductDeriv(sig, v=u)

        if adjoint:
            return 1j * omega(freq) * ( dsig_dm.T * ( dMe_dsig.T * v ) )

        return 1j * omega(freq) * ( dMe_dsig * ( dsig_dm * v ) )

    def getRHS(self, freq, backSigma):
        """
            Function to return the right hand side for the system.
            :param float freq: Frequency
            :param numpy.ndarray (nC,) backSigma: Background conductivity model
            :rtype: numpy.ndarray (nE, 2)
            :return: one RHS for both polarizations
        """
        # Get sources for the frequency
        src = self.survey.getSources(freq)
        # Make sure that there is 2 polarizations. 
        # assert len()
        # Get the background electric fields
        from simpegMT.Sources import homo1DModelSource
        eBG_bp = homo1DModelSource(self.mesh,freq,backSigma)
        Abg = self.getAbg(freq)

        return -Abg*eBG_bp

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
