from SimPEG.EM.Utils.EMUtils import omega, mu_0
from SimPEG import SolverLU as SimpegSolver, PropMaps, Utils, mkvc, sp, np
from SimPEG.EM.FDEM.ProblemFDEM import BaseFDEMProblem
from SurveyNSEM import Survey, Data
from FieldsNSEM import BaseNSEMFields, Fields1D_ePrimSec, Fields3D_ePrimSec
from SimPEG.NSEM.Utils.MT1Danalytic import getEHfields
import time, sys

class BaseNSEMProblem(BaseFDEMProblem):
    """
        Base class for all Natural source problems.
    """

    def __init__(self, mesh, **kwargs):
        BaseFDEMProblem.__init__(self, mesh, **kwargs)
        Utils.setKwargs(self, **kwargs)
    # Set the default pairs of the problem
    surveyPair = Survey
    dataPair = Data
    fieldsPair = BaseNSEMFields

    # Set the solver
    Solver = SimpegSolver
    solverOpts = {}

    verbose = False
    # Notes:
    # Use the forward and devs from BaseFDEMProblem
    # Might need to add more stuff here.

    ## NEED to clean up the Jvec and Jtvec to use Zero and Identities for None components.
    def Jvec(self, m, v, f=None):
        """
        Function to calculate the data sensitivities dD/dm times a vector.

            :param numpy.ndarray m (nC, 1) - conductive model
            :param numpy.ndarray v (nC, 1) - random vector
            :param NSEMfields object (optional) - NSEM fields object, if not given it is calculated
            :rtype: NSEMdata object
            :return: Data sensitivities wrt m
        """

        # Calculate the fields
        if f is None:
           f= self.fields(m)
        # Set current model
        self.curModel = m
        # Initiate the Jv object
        Jv = self.dataPair(self.survey)

        # Loop all the frequenies
        for freq in self.survey.freqs:
            dA_du = self.getA(freq) #

            dA_duI = self.Solver(dA_du, **self.solverOpts)

            for src in self.survey.getSrcByFreq(freq):
                # We need fDeriv_m = df/du*du/dm + df/dm
                # Construct du/dm, it requires a solve
                # NOTE: need to account for the 2 polarizations in the derivatives.
                u_src = f[src,:] # u should be a vector by definition. Need to fix this...
                # dA_dm and dRHS_dm should be of size nE,2, so that we can multiply by dA_duI. The 2 columns are each of the polarizations.
                dA_dm = self.getADeriv_m(freq, u_src, v) # Size: nE,2 (u_px,u_py) in the columns.
                dRHS_dm = self.getRHSDeriv_m(freq, v) # Size: nE,2 (u_px,u_py) in the columns.
                if dRHS_dm is None:
                    du_dm = dA_duI * ( -dA_dm )
                else:
                    du_dm = dA_duI * ( -dA_dm + dRHS_dm )
                # Calculate the projection derivatives
                for rx in src.rxList:
                    # Get the projection derivative
                    # v should be of size 2*nE (for 2 polarizations)
                    PDeriv_u = lambda t: rx.evalDeriv(src, self.mesh, f, t) # wrt u, we don't have have PDeriv wrt m
                    Jv[src, rx] = PDeriv_u(mkvc(du_dm))
            dA_duI.clean()
        # Return the vectorized sensitivities
        return mkvc(Jv)

    def Jtvec(self, m, v, f=None):
        """
        Function to calculate the transpose of the data sensitivities (dD/dm)^T times a vector.

            :param numpy.ndarray m (nC, 1) - conductive model
            :param numpy.ndarray v (nD, 1) - vector
            :param NSEMfields object f (optional) - NSEM fields object, if not given it is calculated
            :rtype: NSEMdata object
            :return: Data sensitivities wrt m
        """

        if f is None:
            f = self.fields(m)

        self.curModel = m

        # Ensure v is a data object.
        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        Jtv = np.zeros(m.size)

        for freq in self.survey.freqs:
            AT = self.getA(freq).T

            ATinv = self.Solver(AT, **self.solverOpts)

            for src in self.survey.getSrcByFreq(freq):
                ftype = self._solutionType
                f_src = f[src, :] # Need to fix this...

                for rx in src.rxList:
                    # Get the adjoint evalDeriv
                    # PTv needs to be nE,
                    PTv = rx.evalDeriv(src, self.mesh, f, mkvc(v[src, rx],2), adjoint=True) # wrt u, need possibility wrt m
                    # Get the
                    dA_duIT = ATinv * PTv
                    dA_dmT = self.getADeriv_m(freq, f_src, mkvc(dA_duIT), adjoint=True)
                    dRHS_dmT = self.getRHSDeriv_m(freq, mkvc(dA_duIT), adjoint=True)
                    # Make du_dmT
                    if dRHS_dmT is None:
                        du_dmT = -dA_dmT
                    else:
                        du_dmT = -dA_dmT + dRHS_dmT
                    # Select the correct component
                    # du_dmT needs to be of size nC,
                    real_or_imag = rx.component
                    if real_or_imag == 'real':
                        Jtv +=  du_dmT.real
                    elif real_or_imag == 'imag':
                        Jtv +=  -du_dmT.real
                    else:
                        raise Exception('Must be real or imag')
            # Clean the factorization, clear memory.
            ATinv.clean()
        return Jtv

###################################
## 1D problems
###################################

class Problem1D_ePrimSec(BaseNSEMProblem):
    """
    A NSEM problem soving a e formulation and primary/secondary fields decomposion.

    By eliminating the magnetic flux density using

        .. math ::

            \mathbf{b} = \\frac{1}{i \omega}\\left(-\mathbf{C} \mathbf{e} \\right)


    we can write Maxwell's equations as a second order system in \\\(\\\mathbf{e}\\\) only:

    .. math ::
        \\left(\mathbf{C}^T \mathbf{M^e_{\mu^{-1}}} \mathbf{C} + i \omega \mathbf{M^f_\sigma}] \mathbf{e}_{s} =& i \omega \mathbf{M^f_{\delta \sigma}} \mathbf{e}_{p}
    which we solve for \\\(\\\mathbf{e_s}\\\). The total field \\\mathbf{e}\\ = \\\mathbf{e_p}\\ + \\\mathbf{e_s}\\.

    The primary field is estimated from a background model (commonly half space ).


    """

    # From FDEMproblem: Used to project the fields. Currently not used for NSEMproblem.
    _solutionType = 'e_1dSolution'
    _formulation  = 'EF'
    fieldsPair = Fields1D_ePrimSec

    # Initiate properties
    _sigmaPrimary = None


    def __init__(self, mesh, **kwargs):
        BaseNSEMProblem.__init__(self, mesh, **kwargs)
        # self._sigmaPrimary = sigmaPrimary
    @property
    def MeMui(self):
        """
            Edge inner product matrix
        """
        if getattr(self, '_MeMui', None) is None:
            self._MeMui = self.mesh.getEdgeInnerProduct(1.0/mu_0)
        return self._MeMui

    @property
    def MfSigma(self):
        """
            Edge inner product matrix
        """
        if getattr(self, '_MfSigma', None) is None:
            self._MfSigma = self.mesh.getFaceInnerProduct(self.curModel.sigma)
        return self._MfSigma

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
            Function to get the A matrix.

            :param float freq: Frequency
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """

        # Note: need to use the code above since in the 1D problem I want
        # e to live on Faces(nodes) and h on edges(cells). Might need to rethink this
        # Possible that _fieldType and _eqLocs can fix this
        MeMui = self.MeMui
        MfSigma = self.MfSigma
        C = self.mesh.nodalGrad
        # Make A
        A = C.T*MeMui*C + 1j*omega(freq)*MfSigma
        # Either return full or only the inner part of A
        return A

    def getADeriv_m(self, freq, u, v, adjoint=False):
        """
        The derivative of A wrt sigma
        """

        dsig_dm = self.curModel.sigmaDeriv
        MeMui = self.MeMui
        #
        u_src = u['e_1dSolution']
        dMfSigma_dm = self.mesh.getFaceInnerProductDeriv(self.curModel.sigma)(u_src) * self.curModel.sigmaDeriv
        if adjoint:
            return 1j * omega(freq) * (  dMfSigma_dm.T * v )
        # Note: output has to be nN/nF, not nC/nE.
        # v should be nC
        return 1j * omega(freq) * ( dMfSigma_dm * v )

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
        # Make the fields object
        F = self.fieldsPair(self.mesh, self.survey)
        # Loop over the frequencies
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

# Note this is not fully functional.
# Missing:
# Fields class corresponding to the fields
# Update Jvec and Jtvec to include all the derivatives components
# Other things ...
class Problem1D_eTotal(BaseNSEMProblem):
    """
    A NSEM problem solving a e formulation and a Total bondary domain decompostion.

    Solves the equation:

    Math:
        Have to do this...
        Not implement correctly.......
    """

    # From FDEMproblem: Used to project the fields. Currently not used for NSEMproblem.
    _solutionType = 'e_1dSolution'
    _formulation  = 'EF'
    # fieldsPair = Fields1D_eTotal

    def __init__(self, mesh, **kwargs):
        BaseNSEMProblem.__init__(self, mesh, **kwargs)
    @property
    def MeMui(self):
        """
            Edge inner product matrix
        """
        if getattr(self, '_MeMui', None) is None:
            self._MeMui = self.mesh.getEdgeInnerProduct(1.0/mu_0)
        return self._MeMui

    @property
    def MfSigma(self):
        """
            Edge inner product matrix
        """
        if getattr(self, '_MfSigma', None) is None:
            self._MfSigma = self.mesh.getFaceInnerProduct(self.curModel.sigma)
        return self._MfSigma

    def getA(self, freq, full=False):
        """
            Function to get the A matrix.

            :param float freq: Frequency
            :param logic full: Return full A or the inner part
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """

        MeMui = self.MeMui
        MfSigma = self.MfSigma
        # Note: need to use the code above since in the 1D problem I want
        # e to live on Faces(nodes) and h on edges(cells). Might need to rethink this
        # Possible that _fieldType and _eqLocs can fix this
        # MeMui = self.MfMui
        # MfSigma = self.MfSigma
        C = self.mesh.nodalGrad
        # Make A
        A = C.T*MeMui*C + 1j*omega(freq)*MfSigma
        # Either return full or only the inner part of A
        if full:
            return A
        else:
            return A[1:-1,1:-1]

    def getADeriv_m(self, freq, u, v, adjoint=False):
        raise NotImplementedError('getADeriv is not implemented')

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

    def getRHSderiv_m(self, freq, backSigma, u, v, adjoint=False):
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

        F = Fields1D_eTotal(self.mesh, self.survey)
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


###################################
## 3D problems
###################################
class Problem3D_ePrimSec(BaseNSEMProblem):
    """
    A NSEM problem solving a e formulation and a primary/secondary fields decompostion.

    By eliminating the magnetic flux density using

        .. math ::

            \mathbf{b} = \\frac{1}{i \omega}\\left(-\mathbf{C} \mathbf{e} \\right)


    we can write Maxwell's equations as a second order system in \\\(\\\mathbf{e}\\\) only:

    .. math ::
        \\left(\mathbf{C}^T \mathbf{M^f_{\mu^{-1}}} \mathbf{C} + i \omega \mathbf{M^e_\sigma}] \mathbf{e}_{s} =& i \omega \mathbf{M^e_{\delta \sigma}} \mathbf{e}_{p}
    which we solve for \\\(\\\mathbf{e_s}\\\). The total field \\\mathbf{e}\\ = \\\mathbf{e_p}\\ + \\\mathbf{e_s}\\.

    The primary field is estimated from a background model (commonly as a 1D model).

    """

    # From FDEMproblem: Used to project the fields. Currently not used for NSEMproblem.
    _solutionType = [ 'e_pxSolution', 'e_pySolution']  # Forces order on the object
    _formulation  = 'EB'
    fieldsPair = Fields3D_ePrimSec

    # Initiate properties
    _sigmaPrimary = None

    def __init__(self, mesh, **kwargs):
        BaseNSEMProblem.__init__(self, mesh, **kwargs)

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
        # Fix u to be a matrix nE,2
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

        F = self.fieldsPair(self.mesh, self.survey)
        for freq in self.survey.freqs:
            if self.verbose:
                startTime = time.time()
                print 'Starting work for {:.3e}'.format(freq)
                sys.stdout.flush()
            A = self.getA(freq)
            rhs  = self.getRHS(freq)
            # Solve the system
            Ainv = self.Solver(A, **self.solverOpts)
            e_s = Ainv * rhs

            # Store the fields
            Src = self.survey.getSrcByFreq(freq)[0]
            # Store the fields
            # Use self._solutionType
            F[Src, 'e_pxSolution'] = e_s[:,0]
            F[Src, 'e_pySolution'] = e_s[:,1]
            # Note curl e = -iwb so b = -curl/iw

            if self.verbose:
                print 'Ran for {:f} seconds'.format(time.time()-startTime)
                sys.stdout.flush()
            Ainv.clean()
        return F
