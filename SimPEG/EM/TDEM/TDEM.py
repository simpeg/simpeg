from SimPEG import Problem, Utils, np, sp, Solver as SimpegSolver
from SimPEG.EM.Base import BaseEMProblem
from SimPEG.EM.TDEM.SurveyTDEM import Survey as SurveyTDEM
from SimPEG.EM.TDEM.FieldsTDEM import * 
from scipy.constants import mu_0 
import time 

class BaseTDEMProblem(Problem.BaseTimeProblem, BaseEMProblem):
    """
    We start with the first order form of Maxwell's equations
    """
    surveyPair = SurveyTDEM
    fieldsPair = Fields

    def __init__(self, mesh, mapping=None, **kwargs):
        Problem.BaseTimeProblem.__init__(self, mesh, mapping=mapping, **kwargs)


    # _FieldsForward_pair = FieldsTDEM  #: used for the forward calculation only

    def fields(self, m):
        """
        Solve the forward problem for the fields.
        
        :param numpy.array m: inversion model (nP,)
        :rtype numpy.array:
        :return F: fields 
        """

        tic = time.time()
        self.curModel = m

        F = self.fieldsPair(self.mesh, self.survey)

        # set initial fields
        for i, src in enumerate(self.survey.srcList):
            F[src,'bSolution',0] = src.bInitial(self)

        # timestep to solve forward
        Ainv = None
        for tInd, dt in enumerate(self.timeSteps):
            print dt, self.timeSteps[tInd]
            if Ainv is not None and (tInd > 0 and dt != self.timeSteps[tInd - 1]):# keep factors if dt is the same as previous step b/c A will be the same  
                Ainv.clean()

            if Ainv is None:
                A = self.getA(tInd)
                if self.verbose: print 'Factoring...   (dt = %e)'%dt
                Ainv = self.Solver(A, **self.solverOpts)
                if self.verbose: print 'Done'

            rhs = self.getRHS(tInd, F)
            if self.verbose: print '    Solving...   (tInd = %d)'%tInd
            sol = Ainv * rhs
            if self.verbose: print '    Done...'
            if sol.ndim == 1:
                sol.shape = (sol.size,1)
            F[:,self._fieldType+'Solution',tInd+1] = sol
        Ainv.clean()
        return F

    def Jvec(self, m, v, u=None):
        raise NotImplementedError

    def Jtvec(self, m, v, u=None):
        raise NotImplementedError

    def getSourceTerm(self, tInd): 
        
        Srcs = self.survey.srcList

        if self._eqLocs is 'FE':
            S_m = np.zeros((self.mesh.nF,len(Srcs)))
            S_e = np.zeros((self.mesh.nE,len(Srcs)))
        elif self._eqLocs is 'EF':
            S_m = np.zeros((self.mesh.nE,len(Srcs)))
            S_e = np.zeros((self.mesh.nF,len(Srcs)))

        for i, src in enumerate(Srcs):
            smi, sei = src.eval(self, self.times[tInd])
            S_m[:,i] = S_m[:,i] + smi
            S_e[:,i] = S_e[:,i] + sei

        return S_m, S_e 



##########################################################################################
################################ E-B Formulation #########################################
##########################################################################################

class Problem_b(BaseTDEMProblem):
    """
    Starting from the quasi-static E-B formulation of Maxwell's equations (semi-discretized) 
    
    .. math::

        \mathbf{C} \mathbf{e} + \\frac{\partial \mathbf{b}}{\partial t} = \mathbf{s_m} \\\\ 
        \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{b} - \mathbf{M_{\sigma}^e} \mathbf{e} = \mathbf{s_e}

    where :math:`\mathbf{s_e}` is an integrated quantity, we eliminate :math:`\mathbf{e}` using 

    .. math::
        \mathbf{e} = \mathbf{M_{\sigma}^e}^{-1} \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{b} - \mathbf{M_{\sigma}^e}^{-1} \mathbf{s_e}

    to obtain a second order semi-discretized system in :math:`\mathbf{b}`

    .. math::
        \mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{b}  + \\frac{\partial \mathbf{b}}{\partial t} = \mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{s_e} + \mathbf{s_m}

    and moving everything except the time derivative to the rhs gives

    .. math::
        \\frac{\partial \mathbf{b}}{\partial t} = -\mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{b} + \mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{s_e} + \mathbf{s_m}

    For the time discretization, we use backward euler. To solve for the :math:`n+1`th time step, we have 

    .. math::
        \\frac{\mathbf{b}^{n+1} - \mathbf{b}^{n}}{\mathbf{dt}} = -\mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{b}^{n+1} + \mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{s_e}^{n+1} + \mathbf{s_m}^{n+1}

    re-arranging to put :math:`\mathbf{b}^{n+1}` on the left hand side gives 

    .. math::
        (\mathbf{I} + \mathbf{dt} \mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f}) \mathbf{b}^{n+1} = \mathbf{b}^{n} + \mathbf{dt}(\mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{s_e}^{n+1} + \mathbf{s_m}^{n+1})

    :param Mesh mesh: mesh
    :param Mapping mapping: mapping
    """

    _fieldType = 'b'
    _eqLocs    = 'FE'
    fieldsPair = Fields_b
    surveyPair = SurveyTDEM

    def __init__(self, mesh, mapping=None, **kwargs):
        BaseTDEMProblem.__init__(self, mesh, mapping=mapping, **kwargs)

    def getA(self, tInd):
        """
        System matrix at a given time index

        .. math::
            (\mathbf{I} + \mathbf{dt} \mathbf{C} \mathbf{M_{\sigma}^e}^{-1} \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f})

        """

        dt = self.timeSteps[tInd]
        C = self.mesh.edgeCurl
        MeSigmaI = self.MeSigmaI
        MfMui = self.MfMui
        I = Utils.speye(self.mesh.nF)

        A = I + dt * ( C * ( MeSigmaI * (C.T * MfMui ) ) )

        if self._makeASymmetric is True:
            return MfMui.T * A
        return A 

    def getADeriv(self, freq, u, v, adjoint=False):
        dt = self.timeSteps[tInd]
        C = self.mesh.edgeCurl
        MeSigmaI = self.MeSigmaIDeriv
        MfMui = self.MfMui
        I = Utils.speye(self.mesh.nF)

        if adjoint:
            if self._makeASymmetric is True:
                v = MfMui * v
            return dt * MfMui.T * ( C * ( MeSigmaIDeriv.T * ( C.T * v ) ) )

        ADeriv = dt * ( C * ( MeSigmaIDeriv * (C.T * ( MfMui * v ) ) ) )
        if self._makeASymmetric is True:
            return MeMui.T * ADeriv
        return ADeriv


    def getRHS(self, tInd, F):
        dt = self.timeSteps[tInd]
        C = self.mesh.edgeCurl
        MeSigmaI = self.MeSigmaI
        MfMui = self.MfMui

        S_m, S_e = self.getSourceTerm(tInd+1) 

        B_n = np.c_[[F[src,'bSolution',tInd] for src in self.survey.srcList]].T
        if B_n.shape[0] is not 1:
            raise NotImplementedError('getRHS not implemented for this shape of B_n')

        rhs = B_n[:,:,0].T + dt * (C * (MeSigmaI * S_e) + S_m)
        if self._makeASymmetric:
            return MfMui.T * rhs
        return rhs

    def getRHSDeriv(self, tInd, F, src, v, adjoint=False):
        
        raise NotImplementedError

        dt = self.timeSteps[tInd]
        C = self.mesh.edgeCurl
        MeSigmaI = self.MeSigmaI
        MeSigmaIDeriv = self.MeSigmaIDeriv
        MfMui = self.MfMui

        S_m, S_e = src.eval(tInd+1, self) # I think this is tInd+1 ? 
        S_m, S_e = src.evalDeriv(tInd+1, self, adjoint=adjoint) # I think this is tInd+1 ? 

        B_n = np.c_[[F[src,'b',tInd] for src in self.survey.srcList]].T
        if B_n.shape[0] is not 1:
            raise NotImplementedError('getRHS not implemented for this shape of B_n')

        # return B_n + dt * (C * (MeSigmaIDeriv * S_e) + S_m)











