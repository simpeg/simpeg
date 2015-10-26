from simpegEM.Utils.EMUtils import omega
from SimPEG import mkvc
from scipy.constants import mu_0
from simpegMT.BaseMT import BaseMTProblem
from simpegMT.SurveyMT import SurveyMT
from simpegMT.FieldsMT import FieldsMT_1D
from simpegMT.DataMT  import DataMT
from simpegMT.Utils.MT1Danalytic import getEHfields
import numpy as np
import multiprocessing, sys, time


class eForm_psField(BaseMTProblem):
    """
    A MT problem soving a e formulation and primary/secondary fields decomposion.

    Solves the equation

    """
    # From FDEMproblem: Used to project the fields. Currently not used for MTproblem.
    _fieldType = 'e_1d'
    _eqLocs    = 'EF'
    _sigmaPrimary = None


    def __init__(self, mesh, **kwargs):
        BaseMTProblem.__init__(self, mesh, **kwargs)
        self.fieldsPair = FieldsMT_1D
        # self._sigmaPrimary = sigmaPrimary

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

        Mmui = self.mesh.getEdgeInnerProduct(1.0/mu_0)
        Msig = self.mesh.getFaceInnerProduct(self.curModel.sigma)
        # Note: need to use the code above since in the 1D problem I want
        # e to live on Faces(nodes) and h on edges(cells). Might need to rethink this
        # Possible that _fieldType and _eqLocs can fix this
        # Mmui = self.MfMui
        # Msig = self.MeSigma
        C = self.mesh.nodalGrad
        # Make A
        A = C.T*Mmui*C + 1j*omega(freq)*Msig
        # Either return full or only the inner part of A
        return A

    def getADeriv_m(self, freq, u, v, adjoint=False):
        """
        The derivative of A wrt sigma
        """

        dsig_dm = self.curModel.sigmaDeriv
        MeMui = self.mesh.getEdgeInnerProduct(1.0/mu_0)
        # Need to make the dMf_dsig symmetirc (nN,nN), don't know how to do this
        dMf_dsig = self.mesh.getFaceInnerProductDeriv(self.curModel.sigma)(u) * self.curModel.sigmaDeriv
        if adjoint:
            return 1j * omega(freq) * (  dMf_dsig.T * v )
        # Note: output has to be nN/nF, not nC/nE.
        # v should be nC
        return 1j * omega(freq) * ( dMf_dsig * v )

    def getRHS(self, freq):
        """
            Function to return the right hand side for the system.
            :param float freq: Frequency
            :rtype: numpy.ndarray (nF, 1), numpy.ndarray (nF, 1)
            :return: RHS for 1 polarizations, primary fields
        """

        # Get sources for the frequncy(polarizations)
        Src = self.survey.getSrcByFreq(freq)[0]
        S_e = Src.S_e(self)
        return -1j * omega(freq) * S_e

    def getRHSDeriv_m(self, freq, v, adjoint=False):
        """
        The derivative of the RHS wrt sigma
        """

        Src = self.survey.getSrcByFreq(freq)[0]
        S_eDeriv = Src.S_eDeriv_m(self, v, adjoint)
        return -1j * omega(freq) * S_eDeriv

    def fields(self, m):
        '''
        Function to calculate all the fields for the model m.

        :param np.ndarray (nC,) m: Conductivity model
        '''
        # Set the current model
        self.curModel = m

        F = FieldsMT_1D(self.mesh, self.survey)
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
            # NOTE: only store the e_solution(secondary), all other components calculated in the fields object
            F[Src, 'e_1dSolution'] = e_s[:,-1] # Only storing the yx polarization as 1d

            # Note curl e = -iwb so b = -curl e /iw
            # b = -( self.mesh.nodalGrad * e )/( 1j*omega(freq) )
            # F[Src, 'b_1d'] = b[:,1]
            if self.verbose:
                print 'Ran for {:f} seconds'.format(time.time()-startTime)
                sys.stdout.flush()
        return F

class eForm_TotalField(BaseMTProblem):
    """
    A MT problem solving a e formulation and a Total bondary domain decompostion.

    Solves the equation:

    Math:


    """

    # From FDEMproblem: Used to project the fields. Currently not used for MTproblem.
    _fieldType = 'e'
    _eqLocs    = 'EF'


    def __init__(self, mesh, **kwargs):
        BaseMTProblem.__init__(self, mesh, **kwargs)

    def getA(self, freq, full=False):
        """
            Function to get the A matrix.

            :param float freq: Frequency
            :param logic full: Return full A or the inner part
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """

        Mmui = self.mesh.getEdgeInnerProduct(1.0/mu_0)
        Msig = self.mesh.getFaceInnerProduct(self.curModel.sigma)
        # Note: need to use the code above since in the 1D problem I want
        # e to live on Faces(nodes) and h on edges(cells). Might need to rethink this
        # Possible that _fieldType and _eqLocs can fix this
        # Mmui = self.MfMui
        # Msig = self.MeSigma
        C = self.mesh.nodalGrad
        # Make A
        A = C.T*Mmui*C + 1j*omega(freq)*Msig
        # Either return full or only the inner part of A
        if full:
            return A
        else:
            return A[1:-1,1:-1]

    def getADeriv(self, freq, u, v, adjoint=False):
        sig = self.curTModel
        dsig_dm = self.curTModelDeriv
        dMe_dsig = self.mesh.getEdgeInnerProductDeriv(sig, v=u)

        if adjoint:
            return 1j * omega(freq) * ( dsig_dm.T * ( dMe_dsig.T * v ) )

        return 1j * omega(freq) * ( dMe_dsig * ( dsig_dm * v ) )

    def getRHS(self, freq):
        """
            Function to return the right hand side for the system.
            :param float freq: Frequency
            :rtype: numpy.ndarray (nE, 2), numpy.ndarray (nE, 2)
            :return: RHS for both polarizations, primary fields
        """
        # Get sources for the frequency
        # NOTE: Need to use the source information, doesn't really apply in 1D
        src = self.survey.getSrcByFreq(freq)
        # Get the full A
        A = self.getA(freq,full=True)
        # Define the outer part of the solution matrix
        Aio = A[1:-1,[0,-1]]
        Ed, Eu, Hd, Hu = getEHfields(self.mesh,self.curModel.sigma,freq,self.mesh.vectorNx)
        Etot = (Ed + Eu)
        sourceAmp = 1.0
        Etot = ((Etot/Etot[-1])*sourceAmp) # Scale the fields to be equal to sourceAmp at the top
        ## Note: The analytic solution is derived with e^iwt
        eBC = np.r_[Etot[0],Etot[-1]]
        # The right hand side

        return -Aio*eBC, eBC

    def getRHSderiv(self, freq, backSigma, u, v, adjoint=False):
        raise NotImplementedError('getRHSDeriv not implemented yet')
        return None

    def fields(self, m):
        '''
        Function to calculate all the fields for the model m.

        :param np.ndarray (nC,) m: Conductivity model
        :param np.ndarray (nC,) m_back: Background conductivity model
        '''
        self.curModel = m
        # RHS, CalcFields = self.getRHS(freq,m_back), self.calcFields

        F = FieldsMT_1D(self.mesh, self.survey)
        for freq in self.survey.freqs:
            if self.verbose:
                startTime = time.time()
                print 'Starting work for {:.3e}'.format(freq)
                sys.stdout.flush()
            A = self.getA(freq)
            rhs, e_o = self.getRHS(freq)
            Ainv = self.Solver(A, **self.solverOpts)
            e_i = Ainv * rhs
            e = mkvc(np.r_[e_o[0], e_i, e_o[1]],2)
            # Store the fields
            Src = self.survey.getSrcByFreq(freq)
            # NOTE: only store e fields
            F[Src, 'e_1dSolution'] = e[:,0]
            if self.verbose:
                print 'Ran for {:f} seconds'.format(time.time()-startTime)
                sys.stdout.flush()
        return F
