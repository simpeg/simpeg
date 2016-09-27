from __future__ import print_function
from SimPEG import Survey, Problem, Utils, Models, np, sp, mkvc, SolverLU as SimpegSolver
from SimPEG.EM.Utils import omega
from scipy.constants import mu_0
from SimPEG.MT.BaseMT import BaseMTProblem
from SimPEG.MT.SurveyMT import Survey, Data
from SimPEG.MT.FieldsMT import Fields3D_e
import multiprocessing, sys, time



class eForm_ps(BaseMTProblem):
    """
    A MT problem solving a e formulation and a primary/secondary fields decompostion.

    By eliminating the magnetic flux density using

        .. math ::

            \mathbf{b} = \\frac{1}{i \omega}\\left(-\mathbf{C} \mathbf{e} \\right)


    we can write Maxwell's equations as a second order system in \\\(\\\mathbf{e}\\\) only:

    .. math ::
        \\left(\mathbf{C}^T \mathbf{M^f_{\mu^{-1}}} \mathbf{C} + i \omega \mathbf{M^e_\sigma}] \mathbf{e}_{s} =& i \omega \mathbf{M^e_{\delta \sigma}} \mathbf{e}_{p}
    which we solve for \\\(\\\mathbf{e_s}\\\). The total field \\\mathbf{e}\\ = \\\mathbf{e_p}\\ + \\\mathbf{e_s}\\.

    The primary field is estimated from a background model (commonly as a 1D model).

    """

    # From FDEMproblem: Used to project the fields. Currently not used for MTproblem.
    _fieldType = 'e'
    _eqLocs    = 'FE'
    fieldsPair = Fields3D_e
    _sigmaPrimary = None

    def __init__(self, mesh, **kwargs):
        BaseMTProblem.__init__(self, mesh, **kwargs)

    @property
    def sigmaPrimary(self):
        """
        A background model, use for the calculation of the primary fields.

        """
        return self._sigmaPrimary
    @sigmaPrimary.setter
    def sigmaPrimary(self, val):
        # Note: TODO add logic for val, make sure it is the correct size.
        self._sigmaPrimary = val

    def getA(self, freq):
        """
            Function to get the A system.

            :param float freq: Frequency
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """
        Mmui = self.MfMui
        Msig = self.MeSigma
        C = self.mesh.edgeCurl

        return C.T*Mmui*C + 1j*omega(freq)*Msig

    def getADeriv_m(self, freq, u, v, adjoint=False):
        """
        Calculate the derivative of A wrt m.

        """

        # This considers both polarizations and returns a nE,2 matrix for each polarization
        if adjoint:
            dMe_dsigV = sp.hstack(( self.MeSigmaDeriv( u['e_pxSolution'] ).T, self.MeSigmaDeriv(u['e_pySolution'] ).T ))*v
        else:
            # Need a nE,2 matrix to be returned
            dMe_dsigV = np.hstack(( mkvc(self.MeSigmaDeriv( u['e_pxSolution'] )*v,2), mkvc( self.MeSigmaDeriv(u['e_pySolution'] )*v,2) ))
        return 1j * omega(freq) * dMe_dsigV


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
        S_eDeriv = Src.S_eDeriv_m(self, v, adjoint)
        return -1j * omega(freq) * S_eDeriv

    def fields(self, m):
        '''
        Function to calculate all the fields for the model m.

        :param np.ndarray (nC,) m: Conductivity model
        '''
        # Set the current model
        self.curModel = m

        F = Fields3D_e(self.mesh, self.survey)
        for freq in self.survey.freqs:
            if self.verbose:
                startTime = time.time()
                print('Starting work for {:.3e}'.format(freq))
                sys.stdout.flush()
            A = self.getA(freq)
            rhs  = self.getRHS(freq)
            # Solve the system
            Ainv = self.Solver(A, **self.solverOpts)
            e_s = Ainv * rhs

            # Store the fields
            Src = self.survey.getSrcByFreq(freq)[0]
            # Store the fieldss
            F[Src, 'e_pxSolution'] = e_s[:,0]
            F[Src, 'e_pySolution'] = e_s[:,1]
            # Note curl e = -iwb so b = -curl/iw

            if self.verbose:
                print('Ran for {:f} seconds'.format(time.time()-startTime))
                sys.stdout.flush()
            Ainv.clean()
        return F

