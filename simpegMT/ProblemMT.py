from SimPEG import Survey, Problem, Utils, Models, np, sp, SolverLU as SimpegSolver
from simpegEM.FDEM import BaseFDEMProblem
from simpegEM.Utils import omega
from scipy.constants import mu_0
from SurveyMT import SurveyMT, FieldsMT
import multiprocessing, sys, time



class BaseMTProblem(BaseFDEMProblem):

    def __init__(self, mesh, **kwargs):
        BaseFDEMProblem.__init__(self, mesh, **kwargs)


    surveyPair = SurveyMT
    dataPair = Survey.Data

    Solver = SimpegSolver
    solverOpts = {}

    verbose = False
    # Notes:
    # Use the forward and devs from BaseFDEMProblem
    # Might need to add more stuff here.


    @property
    def MeSigmaBG(self):
        #TODO: hardcoded to sigma as the model
        if getattr(self, '_MeSigmaBG', None) is None:
            sigmaBG = self.backModel
            self._MeSigmaBG = self.mesh.getEdgeInnerProduct(sigmaBG)
        return self._MeSigmaBG

    
class ProblemMT_eForm_ps(BaseMTProblem):
    """ 
    A MT problem solving a e formulation and a primary/secondary fields decompostion.

    Solves the equation
    """

    _fieldType = 'e'
    _eqLocs    = 'FE'
    fieldsPair = FieldsMT
    # Set new properties 
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

    def __init__(self, mesh, **kwargs):
        BaseMTProblem.__init__(self, mesh, **kwargs)

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
        deltM = self.curModel - self.backModel
        Abg = -1j*omega(freq)*deltM*eBG_bp

        return Abg*eBG_bp, eBG_bp
    def getRHSderiv(self, freq, backSigma, u, v, adjoint=False):
        raise NotImplementedError('getRHSDeriv not implemented yet')
        return None

    def fields(self, m, m_back):
        '''
        Function to calculate all the fields for the model m.

        :param np.ndarray (nC,) m: Conductivity model
        :param np.ndarray (nC,) m_back: Background conductivity model
        '''
        self.curModel = m
        self.backModel = m_back
        # RHS, CalcFields = self.getRHS(freq,m_back), self.calcFields

        F = FieldsMT(self.mesh, self.survey)

        if verbose:
            startTime = time.time()
            print 'Starting work for {:.3e}'.format(freq)
            sys.stdout.flush()
        A = self.getA(freq)
        rhs, e_p = self.getRHS(freq,m_back)
        Ainv = self.Solver(A, **self.solverOpts)
        e_s = Ainv * rhs 
        e = e_p + e_s
        # Store the fields
        Src = self.survey.getSources(freq)
        # Store the fieldss
        F[Src, 'e_px'] = e[:,0]
        F[Src, 'e_py'] = e[:,1]
        # Note curl e = -iwb so b = -curl/iw
        b = -( self.mesh.edgeCurl * e )/( 1j*omega(freq) )
        F[Src, 'b_px'] = b[:,0]
        F[Src, 'b_py'] = b[:,1]
        if verbose:
            print 'Ran for {:f} seconds'.format(time.time()-startTime)
            sys.stdout.flush()
        return F
        

