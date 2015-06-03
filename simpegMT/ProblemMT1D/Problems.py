from simpegEM.Utils.EMUtils import omega
from SimPEG import mkvc
from scipy.constants import mu_0
from simpegMT.BaseMT import BaseMTProblem
from simpegMT.SurveyMT import SurveyMT
from simpegMT.FieldsMT import FieldsMT
from simpegMT.DataMT  import DataMT
from simpegMT.Utils.MT1Danalytic import getEHfields
import numpy as np
import multiprocessing, sys, time


class eForm_TotalField(BaseMTProblem):
    """ 
    A MT problem solving a e formulation and a primary/secondary fields decompostion.

    Solves the equation:


    """

    # From FDEMproblem: Used to project the fields. Currently not used for MTproblem.
    _fieldType = 'e'
    _eqLocs    = 'FE'
    

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
        Msig = self.mesh.getFaceInnerProduct(self.curModel)
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
        src = self.survey.getSources(freq)
        # Get the full A
        A = self.getA(freq,full=True)
        # Define the outer part of the solution matrix
        Aio = A[1:-1,[0,-1]]
        Ed, Eu, Hd, Hu = getEHfields(self.mesh,self.curModel,freq,self.mesh.vectorNx)
        Etot = (Ed + Eu)
        sourceAmp = 1.0
        Etot = ((Etot/Etot[-1])*sourceAmp) # Scale the fields to be equal to sourceAmp at the top
        ## Note: The analytic solution is derived with e^iwt
        eBC = np.r_[Etot[0],Etot[-1]]
        # The right hand side

        return Aio*eBC, eBC

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

        F = FieldsMT(self.mesh, self.survey)
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
            Src = self.survey.getSources(freq)
            # Store the fields
            # NOTE: only store 
            F[Src, 'e_1d'] = e
            # F[Src, 'e_py'] = 0*e[:,0]
            # Note curl e = -iwb so b = -curl e /iw
            b = -( self.mesh.nodalGrad * e )/( 1j*omega(freq) )
            # F[Src, 'b_px'] = 0*b[:,0]
            F[Src, 'b_1d'] = b
            if self.verbose:
                print 'Ran for {:f} seconds'.format(time.time()-startTime)
                sys.stdout.flush()
        return F
        