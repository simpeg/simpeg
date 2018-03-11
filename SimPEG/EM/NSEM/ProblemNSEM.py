from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import time
import sys
import scipy.sparse as sp
import numpy as np

from SimPEG.EM.Utils.EMUtils import omega, mu_0
from SimPEG import SolverLU as SimpegSolver, Utils, mkvc
from ..FDEM.ProblemFDEM import BaseFDEMProblem
from .SurveyNSEM import Survey, Data
from .FieldsNSEM import BaseNSEMFields, Fields1D_ePrimSec, Fields3D_ePrimSec


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
    # Use the fields and devs methods from BaseFDEMProblem

    # NEED to clean up the Jvec and Jtvec to use Zero and Identities for None components.
    def Jvec(self, m, v, f=None):
        """
        Function to calculate the data sensitivities dD/dm times a vector.

        :param numpy.ndarray m: conductivity model (nP,)
        :param numpy.ndarray v: vector which we take sensitivity product with (nP,)
        :param SimPEG.EM.NSEM.FieldsNSEM (optional) u: NSEM fields object, if not given it is calculated
        :rtype: numpy.ndarray
        :return: Jv (nData,) Data sensitivities wrt m
        """

        # Calculate the fields if not given as input
        if f is None:
           f = self.fields(m)
        # Set current model
        self.model = m
        # Initiate the Jv object
        Jv = self.dataPair(self.survey)

        # Loop all the frequenies
        for freq in self.survey.freqs:
            # Get the system
            A = self.getA(freq)
            # Factor
            Ainv = self.Solver(A, **self.solverOpts)

            for src in self.survey.getSrcByFreq(freq):
                # We need fDeriv_m = df/du*du/dm + df/dm
                # Construct du/dm, it requires a solve
                # NOTE: need to account for the 2 polarizations in the derivatives.
                u_src = f[src,:] # u should be a vector by definition. Need to fix this...
                # dA_dm and dRHS_dm should be of size nE,2, so that we can multiply by Ainv.
                # The 2 columns are each of the polarizations.
                dA_dm_v = self.getADeriv(freq, u_src, v) # Size: nE,2 (u_px,u_py) in the columns.
                dRHS_dm_v = self.getRHSDeriv(freq, v) # Size: nE,2 (u_px,u_py) in the columns.
                # Calculate du/dm*v
                du_dm_v = Ainv * ( - dA_dm_v + dRHS_dm_v)
                # Calculate the projection derivatives
                for rx in src.rxList:
                    # Calculate dP/du*du/dm*v
                    Jv[src, rx] = rx.evalDeriv(src, self.mesh, f, mkvc(du_dm_v)) # wrt uPDeriv_u(mkvc(du_dm))
            Ainv.clean()
        # Return the vectorized sensitivities
        return mkvc(Jv)

    def Jtvec(self, m, v, f=None):
        """
        Function to calculate the transpose of the data sensitivities (dD/dm)^T times a vector.

        :param numpy.ndarray m: inversion model (nP,)
        :param numpy.ndarray v: vector which we take adjoint product with (nP,)
        :param SimPEG.EM.NSEM.FieldsNSEM f (optional): NSEM fields object, if not given it is calculated
        :rtype: numpy.ndarray
        :return: Jtv (nP,) Data sensitivities wrt m
        """

        if f is None:
            f = self.fields(m)

        self.model = m

        # Ensure v is a data object.
        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        Jtv = np.zeros(m.size)

        for freq in self.survey.freqs:
            AT = self.getA(freq).T

            ATinv = self.Solver(AT, **self.solverOpts)

            for src in self.survey.getSrcByFreq(freq):
                # u_src needs to have both polarizations
                u_src = f[src, :]

                for rx in src.rxList:
                    # Get the adjoint evalDeriv
                    # PTv needs to be nE,2
                    PTv = rx.evalDeriv(src, self.mesh, f, mkvc(v[src, rx]), adjoint=True) # wrt f, need possibility wrt m
                    # Get the
                    dA_duIT = mkvc(ATinv * PTv) # Force (nU,) shape
                    dA_dmT = self.getADeriv(freq, u_src, dA_duIT, adjoint=True)
                    dRHS_dmT = self.getRHSDeriv(freq, dA_duIT, adjoint=True)
                    # Make du_dmT
                    du_dmT = -dA_dmT + dRHS_dmT
                    # Select the correct component
                    # du_dmT needs to be of size (nP,) number of model parameters
                    real_or_imag = rx.component
                    if real_or_imag == 'real':
                        Jtv +=  np.array(du_dmT, dtype=complex).real
                    elif real_or_imag == 'imag':
                        Jtv +=  -np.array(du_dmT, dtype=complex).real
                    else:
                        raise Exception('Must be real or imag')
            # Clean the factorization, clear memory.
            ATinv.clean()
        return Jtv

###################################
# 1D problems
###################################


class Problem1D_ePrimSec(BaseNSEMProblem):
    """
    A NSEM problem soving a e formulation and primary/secondary fields decomposion.

    By eliminating the magnetic flux density using

        .. math ::

            \mathbf{b} = \\frac{1}{i \omega}\\left(-\mathbf{C} \mathbf{e} \\right)


    we can write Maxwell's equations as a second order system in \\\(\\\mathbf{e}\\\) only:

    .. math ::
        \\left[ \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^e } \mathbf{C} + i \omega \mathbf{M_{\sigma}^f} \\right] \mathbf{e}_{s} = i \omega \mathbf{M_{\sigma_{s}}^f } \mathbf{e}_{p}

    which we solve for :math:`\\mathbf{e_s}`. The total field :math:`\mathbf{e} = \mathbf{e_p} + \mathbf{e_s}`.

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
        # if getattr(self, '_MfSigma', None) is None:
        self._MfSigma = self.mesh.getFaceInnerProduct(self.sigma)
        return self._MfSigma

    def MfSigmaDeriv(self, u):
        """
            Edge inner product matrix
        """
        # if getattr(self, '_MfSigmaDeriv', None) is None:
        self._MfSigmaDeriv = self.mesh.getFaceInnerProductDeriv(self.sigma)(u) * self.sigmaDeriv
        return self._MfSigmaDeriv

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

    def getADeriv(self, freq, u, v, adjoint=False):
        """
        The derivative of A wrt sigma
        """

        u_src = u['e_1dSolution']
        dMfSigma_dm = self.MfSigmaDeriv(u_src)
        if adjoint:
            return 1j * omega(freq) * mkvc(dMfSigma_dm.T * v,)
        # Note: output has to be nN/nF, not nC/nE.
        # v should be nC
        return 1j * omega(freq) * mkvc(dMfSigma_dm * v,)

    def getRHS(self, freq):
        """
            Function to return the right hand side for the system.
            :param float freq: Frequency
            :rtype: numpy.ndarray
            :return: RHS for 1 polarizations, primary fields (nF, 1)
        """

        # Get sources for the frequncy(polarizations)
        Src = self.survey.getSrcByFreq(freq)[0]
        # Only select the yx polarization
        S_e = mkvc(Src.S_e(self)[:, 1], 2)
        return -1j * omega(freq) * S_e

    def getRHSDeriv(self, freq, v, adjoint=False):
        """
        The derivative of the RHS wrt sigma
        """

        Src = self.survey.getSrcByFreq(freq)[0]

        S_eDeriv = mkvc(Src.S_eDeriv_m(self, v, adjoint),)
        return -1j * omega(freq) * S_eDeriv

    def fields(self, m=None):
        """
        Function to calculate all the fields for the model m.

        :param numpy.ndarray m: Conductivity model (nC,)
        :rtype: SimPEG.EM.NSEM.FieldsNSEM.Fields1D_ePrimSec
        :return: NSEM fields object containing the solution
        """
        # Set the current model
        if m is not None:
            self.model = m
        # Make the fields object
        F = self.fieldsPair(self.mesh, self.survey)
        # Loop over the frequencies
        for freq in self.survey.freqs:
            if self.verbose:
                startTime = time.time()
                print('Starting work for {:.3e}'.format(freq))
                sys.stdout.flush()
            A = self.getA(freq)
            rhs  = self.getRHS(freq)
            Ainv = self.Solver(A, **self.solverOpts)
            e_s = Ainv * rhs

            # Store the fields
            Src = self.survey.getSrcByFreq(freq)[0]
            # NOTE: only store the e_solution(secondary), all other components calculated in the fields object
            F[Src, 'e_1dSolution'] = e_s

            if self.verbose:
                print('Ran for {:f} seconds'.format(time.time()-startTime))
                sys.stdout.flush()
        return F


###################################
# 3D problems
###################################
class Problem3D_ePrimSec(BaseNSEMProblem):
    """
    A NSEM problem solving a e formulation and a primary/secondary fields decompostion.

    By eliminating the magnetic flux density using

        .. math ::

            \mathbf{b} = \\frac{1}{i \omega}\\left(-\mathbf{C} \mathbf{e} \\right)


    we can write Maxwell's equations as a second order system in :math:`\mathbf{e}` only:

    .. math ::

        \\left[\mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{C} + i \omega \mathbf{M_{\sigma}^e} \\right] \mathbf{e}_{s} = i \omega \mathbf{M_{\sigma_{p}}^e} \mathbf{e}_{p}

    which we solve for :math:`\mathbf{e_s}`. The total field :math:`\mathbf{e} = \mathbf{e_p} + \mathbf{e_s}`.

    The primary field is estimated from a background model (commonly as a 1D model).

    """

    # From FDEMproblem: Used to project the fields. Currently not used for NSEMproblem.
    _solutionType = ['e_pxSolution', 'e_pySolution']  # Forces order on the object
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
        Mfmui = self.MfMui
        Mesig = self.MeSigma
        C = self.mesh.edgeCurl

        return C.T*Mfmui*C + 1j*omega(freq)*Mesig

    def getADeriv(self, freq, u, v, adjoint=False):
        """
        Calculate the derivative of A wrt m.

        :param float freq: Frequency
        :param SimPEG.EM.NSEM.FieldsNSEM u: NSEM Fields object
        :param numpy.ndarray v: vector of size (nU,) (adjoint=False)
            and size (nP,) (adjoint=True)
        :rtype: numpy.ndarray
        :return: Calculated derivative (nP,) (adjoint=False) and (nU,)[NOTE return as a (nU/2,2)
            columnwise polarizations] (adjoint=True) for both polarizations

        """
        # Fix u to be a matrix nE,2
        # This considers both polarizations and returns a nE,2 matrix for each polarization
        # The solution types
        sol0, sol1 = self._solutionType

        if adjoint:
            dMe_dsigV = sp.hstack(( self.MeSigmaDeriv( u[sol0] ).T, self.MeSigmaDeriv(u[sol1] ).T ))*v
        else:
            # Need a nE,2 matrix to be returned
            dMe_dsigV = np.hstack(( mkvc(self.MeSigmaDeriv( u[sol0] )*v, 2), mkvc( self.MeSigmaDeriv(u[sol1] )*v, 2) ))
        return 1j * omega(freq) * dMe_dsigV

    def getRHS(self, freq):
        """
        Function to return the right hand side for the system.

        :param float freq: Frequency
        :rtype: numpy.ndarray
        :return: RHS for both polarizations, primary fields (nE, 2)

        """

        # Get sources for the frequncy(polarizations)
        Src = self.survey.getSrcByFreq(freq)[0]
        S_e = Src.S_e(self)
        return -1j * omega(freq) * S_e

    def getRHSDeriv(self, freq, v, adjoint=False):
        """
        The derivative of the RHS with respect to the model and the source

        :param float freq: Frequency
        :param numpy.ndarray v: vector of size (nU,) (adjoint=False)
            and size (nP,) (adjoint=True)
        :rtype: numpy.ndarray
        :return: Calculated derivative (nP,) (adjoint=False) and (nU,2) (adjoint=True)
            for both polarizations

        """

        # Note: the formulation of the derivative is the same for adjoint or not.
        Src = self.survey.getSrcByFreq(freq)[0]
        S_eDeriv = Src.S_eDeriv(self, v, adjoint)
        dRHS_dm = -1j * omega(freq) * S_eDeriv

        return dRHS_dm

    def fields(self, m=None):
        """
        Function to calculate all the fields for the model m.

        :param numpy.ndarray (nC,) m: Conductivity model
        :rtype: SimPEG.EM.NSEM.FieldsNSEM
        :return: Fields object with of the solution

        """
        # Set the current model
        if m is not None:
            self.model = m

        F = self.fieldsPair(self.mesh, self.survey)
        for freq in self.survey.freqs:
            if self.verbose:
                startTime = time.time()
                print('Starting work for {:.3e}'.format(freq))
                sys.stdout.flush()
            A = self.getA(freq)
            rhs = self.getRHS(freq)
            # Solve the system
            Ainv = self.Solver(A, **self.solverOpts)
            e_s = Ainv * rhs

            # Store the fields
            Src = self.survey.getSrcByFreq(freq)[0]
            # Store the fields
            # Use self._solutionType
            F[Src, 'e_pxSolution'] = e_s[:, 0]
            F[Src, 'e_pySolution'] = e_s[:, 1]
            # Note curl e = -iwb so b = -curl/iw

            if self.verbose:
                print('Ran for {:f} seconds'.format(time.time()-startTime))
                sys.stdout.flush()
            Ainv.clean()
        return F
