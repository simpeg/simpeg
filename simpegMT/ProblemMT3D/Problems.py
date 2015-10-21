from SimPEG import Survey, Problem, Utils, Models, np, sp, SolverLU as SimpegSolver
from simpegEM.Utils.EMUtils import omega
from scipy.constants import mu_0
from simpegMT.BaseMT import BaseMTProblem
from simpegMT.SurveyMT import SurveyMT
from simpegMT.FieldsMT import FieldsMT_3D
from simpegMT.DataMT import DataMT
import multiprocessing, sys, time



class eForm_ps(BaseMTProblem):
    """
    A MT problem solving a e formulation and a primary/secondary fields decompostion.

    Solves the equation:



    """

    # From FDEMproblem: Used to project the fields. Currently not used for MTproblem.
    _fieldType = 'e_px'
    _eqLocs    = 'FE'
    fieldsPair = FieldsMT_3D
    _sigmaPrimary = None

    def __init__(self, mesh, **kwargs):
        BaseMTProblem.__init__(self, mesh, **kwargs)

    @property
    def sigmaPrimary(self):
        return self._sigmaPrimary
    @sigmaPrimary.setter
    def sigmaPrimary(self, val):
        # Note: TODO add logic for val, make sure it is the correct size.
        self._sigmaPrimary = val

    def getA(self, freq):
        """
            Function to get the A matrix.

            :param float freq: Frequency
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """
        Mmui = self.MfMui
        Msig = self.MeSigma
        C = self.mesh.edgeCurl

        return C.T*Mmui*C + 1j*omega(freq)*Msig

    def getADeriv_m(self, freq, u, v, adjoint=False):

        dMe_dsig = self.MeSigmaDeriv( u )

        if adjoint:
            return 1j * omega(freq) * ( dMe_dsig.T * v ) # As in simpegEM
            # return 1j * omega(freq) * ( dsig_dm.T * ( dMe_dsig.T * v ) )

        return 1j * omega(freq) * ( dMe_dsig * v ) # As in simpegEM
        # return 1j * omega(freq) * ( dMe_dsig * ( dsig_dm * v ) )

    def getRHS(self, freq):
        """
            Function to return the right hand side for the system.
            :param float freq: Frequency
            :rtype: numpy.ndarray (nE, 2), numpy.ndarray (nE, 2)
            :return: RHS for both polarizations, primary fields
        """

        # Get sources for the frequncy(polarizations)
        Src = self.survey.getSrcByFreq(freq)[0]
        S_e = Src.S_e(self)
        return -1j * omega(freq) * S_e

    def getRHSDeriv_m(self, freq, v, adjoint=False):
        """
        The derivative of the RHS with respect to sigma
        """

        Src = self.survey.getSrcByFreq(freq)[0]
        S_eDeriv = Src.S_eDeriv(self, v, adjoint)
        return -1j * omega(freq) * S_eDeriv

    def fields(self, m):
        '''
        Function to calculate all the fields for the model m.

        :param np.ndarray (nC,) m: Conductivity model
        '''
        # Set the current model
        self.curModel = m

        F = FieldsMT_3D(self.mesh, self.survey)
        for freq in self.survey.freqs:
            if self.verbose:
                startTime = time.time()
                print 'Starting work for {:.3e}'.format(freq)
                sys.stdout.flush()
            A = self.getA(freq)
            rhs  = self.getRHS(freq)
            Ainv = self.Solver(A, **self.solverOpts)
            e_s = Ainv * rhs

            # Store the fields
            Src = self.survey.getSrcByFreq(freq)[0]
            # Store the fieldss
            F[Src, 'e_pxSolution'] = e_s[:,0]
            F[Src, 'e_pySolution'] = e_s[:,1]
            # Note curl e = -iwb so b = -curl/iw
            # b = -( self.mesh.edgeCurl * e )/( 1j*omega(freq) )
            # F[Src, 'b_px'] = b[:,0]
            # F[Src, 'b_py'] = b[:,1]
            if self.verbose:
                print 'Ran for {:f} seconds'.format(time.time()-startTime)
                sys.stdout.flush()
        return F

class eForm_Tp(BaseMTProblem):
    """
    A MT problem solving a e formulation and a total/primary fields decompostion.

    Solves the equation
    """

    _fieldType = 'e'
    _eqLocs    = 'FE'
    fieldsPair = FieldsMT_3D

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

    @property
    def MeSigmaBack(self):
        #TODO: hardcoded to sigma as the model
        if getattr(self, '_MeSigmaBack', None) is None:
            sigma = self.curModel
            sigmaBG = self.backModel
            self._MeSigmaBack = self.mesh.getEdgeInnerProduct(sigmaBG)
        return self._MeSigmaBack

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
        MeBack = self.MeSigmaBack
        # Set up the A system
        mui = self.MfMui
        C = self.mesh.edgeCurl
        Abg = C.T*mui*C + 1j*omega(freq)*MeBack

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

        F = FieldsMT_3D(self.mesh, self.survey)
        for freq in self.survey.freqs:
            if self.verbose:
                startTime = time.time()
                print 'Starting work for {:.3e}'.format(freq)
                sys.stdout.flush()
            A = self.getA(freq)
            rhs, e_p = self.getRHS(freq,m_back)
            Ainv = self.Solver(A, **self.solverOpts)
            e_s = Ainv * rhs
            e = e_s
            # Store the fields
            Src = self.survey.getSources(freq)
            # Store the fieldss
            F[Src, 'e_px'] = e[:,0]
            F[Src, 'e_py'] = e[:,1]
            # Note curl e = -iwb so b = -curl/iw
            b = -( self.mesh.edgeCurl * e )/( 1j*omega(freq) )
            F[Src, 'b_px'] = b[:,0]
            F[Src, 'b_py'] = b[:,1]
            if self.verbose:
                print 'Ran for {:f} seconds'.format(time.time()-startTime)
                sys.stdout.flush()
        return F

