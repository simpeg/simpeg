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
        F[:,self._fieldType+'Solution',0] = self.getInitialFields()

        # timestep to solve forward
        Ainv = None
        for tInd, dt in enumerate(self.timeSteps):
            if Ainv is not None and (tInd > 0 and dt != self.timeSteps[tInd - 1]):# keep factors if dt is the same as previous step b/c A will be the same  
                Ainv.clean()
                Ainv = None

            if Ainv is None:
                A = self.getAdiag(tInd)
                if self.verbose: print 'Factoring...   (dt = %e)'%dt
                Ainv = self.Solver(A, **self.solverOpts)
                if self.verbose: print 'Done'

            rhs = self.getRHS(tInd)
            Asubdiag = self.getAsubdiag(tInd)
            
            if self.verbose: print '    Solving...   (tInd = %d)'%tInd
            sol = Ainv * (rhs - Asubdiag * F[:,self._fieldType+'Solution',tInd])
            
            if self.verbose: print '    Done...'
            
            if sol.ndim == 1:
                sol.shape = (sol.size,1)
            F[:,self._fieldType+'Solution',tInd+1] = sol

        Ainv.clean()
        return F


    def Jvec(self, m, v, u=None):

        if u is None:
           u = self.fields(m)

        ftype = self._fieldType + 'Solution' # the thing we solved for
        self.curModel = m

        Jv = self.dataPair(self.survey) 

        # mat to store previous time-step's solution deriv times a vector for each source
        # size: nu x nSrc
        dun_dm_v = self.getInitialFieldsDeriv(v) # can over-write this at each timestep

        # 
        df_dm_v = Fields_Derivs(self.mesh, self.survey) # store the field derivs we need to project to calc full deriv
        
        Ainv = None

        for tInd, dt in enumerate(self.timeSteps):
            if Ainv is not None and (tInd > 0 and dt != self.timeSteps[tInd - 1]):# keep factors if dt is the same as previous step b/c A will be the same  
                Ainv.clean()
                Ainv = None

            if Ainv is None:
                A = self.getAdiag(tInd)
                Ainv = self.Solver(A, **self.solverOpts)

            Asubdiag = self.getAsubdiag(tInd)

            for i, src in enumerate(self.survey.srcList): 
                
                un_src = u[src,ftype,tInd+1]

                dA_dm_v   = self.getAdiagDeriv(tInd, un_src, v)
                dRHS_dm_v = self.getRHSDeriv(tInd, src, v)
                # dAsubdiag_dm_v = 0  

                JRHS = dRHS_dm_v - dA_dm_v  # - dAsubdiag_dm_v (which is zero) 

                for rx in src.rxList:
                    df_dmFun = getattr(u, '_%sDeriv'%rx.projField, None)
                    df_dm_v[src, '%sDeriv'%rx.projField , tInd] = df_dmFun(tInd, src, dun_dm_v[:,i], v)

                # step in time and overwrite 
                if tInd != len(self.timeSteps):
                    dun_dm_v[:,i] = Ainv * (JRHS - Asubdiag * dun_dm_v[:,i])

        for src in self.survey.srcList:
            for rx in src.rxList: 
                Jv[src,rx] = rx.evalDeriv(src, self.mesh, self.timeMesh, df_dm_v)

        Ainv.clean()
        return Utils.mkvc(Jv)


    def Jtvec(self, m, v, u=None):

        if u is None:
            u = self.fields(m)

        self.curModel = m
        ftype = self._fieldType + 'Solution' # the thing we solved for

        # Ensure v is a data object.
        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        # TODO: make this general
        if self._fieldType is 'b':
            dun_dmT_v = np.zeros((len(m), self.mesh.nF)) 

        # df_dm_v = Fields_Derivs(self.mesh, self.survey)
        JTv = np.zeros(m.size)
        PT_v = Fields_Derivs(self.mesh, self.survey) #PT_v is a fields object

        for src in self.survey.srcList: 
            for rx in src.rxList: 
                PT_v[src,'%sDeriv'%rx.projField, :] = rx.evalDeriv(src, self.mesh, self.timeMesh, v, adjoint = True) # All the fields for a given src, reciever.
                

        ATinv = None

        for tInd, dt in enumerate(reversed(list(self.timeSteps))): 
            if ATinv is not None and (tInd < self.nT and dt != self.timeSteps[tInd - 1]):# keep factors if dt is the same as previous step b/c A will be the same  
                ATinv.clean()
                ATinv = None

            if ATinv is None:
                A = self.getAdiag(tInd)
                ATinv = self.Solver(A.T, **self.solverOpts)

            for i, src in enumerate(self.survey.srcList):

                u_src = u[src,ftype,tInd+1] # fields for this source at tInd 

                for rx in src.rxList: 

                    df_duTFun = getattr(u, '_%sDeriv'%rx.projField, None)
                    df_duT_v, df_dmT_v = df_duTFun(tInd, src, None, PT_v[src,'%sDeriv'%rx.projField,tInd], adjoint=True)

                    ATinv_df_duT_v = ATinv * df_duT_v
                    rhsT_v = self.getJRHS(tInd, src, u_src, ATinv_df_duT_v, dun_dmT_v[:,i], adjoint = True)

                JTv += rhsT_v + df_dmT_v

        return Utils.mkvc(JTv)

         

    # def getJRHS(self, tInd, src, u, v, adjoint = False): 

    #     dA_dm   = self.getADeriv(tInd, u, v, adjoint)
    #     dRHS_dm = self.getRHSDeriv(tInd, src, v, adjoint)

    #     b = - dA_dm + dRHS_dm

    #     return b
        

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

    def getInitialFields(self):

        Srcs = self.survey.srcList 

        if self._fieldType is 'b' or self._fieldType is 'j':
            ifields = np.zeros((self.mesh.nF, len(Srcs)))
        elif self._fieldType is 'e' or self._fieldType is 'h':
            ifields = np.zeros((self.mesh.nE, len(Srcs)))

        for i,src in enumerate(Srcs): 
            ifields[:,i] = ifields[:,i] + getattr(src, '%sInitial'%self._fieldType, None)(self)

        return ifields

    def getInitialFieldsDeriv(self, v, adjoint=False):
        
        Srcs = self.survey.srcList 

        if self._fieldType is 'b' or self._fieldType is 'j':
            ifieldsDeriv = np.zeros((self.mesh.nF, len(Srcs)))
        elif self._fieldType is 'e' or self._fieldType is 'h':
            ifieldsDeriv = np.zeros((self.mesh.nE, len(Srcs)))

        for i,src in enumerate(Srcs): 
            ifieldsDeriv[:,i] = ifieldsDeriv[:,i] + getattr(src, '%sInitialDeriv'%self._fieldType, None)(self,v,adjoint)

        return ifieldsDeriv


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

    def getAdiag(self, tInd):
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

        A = 1./dt * I + ( C * ( MeSigmaI * (C.T * MfMui ) ) )

        if self._makeASymmetric is True:
            return MfMui.T * A
        return A 

    def getAdiagDeriv(self, tInd, u, v, adjoint=False):
        C = self.mesh.edgeCurl
        MeSigmaIDeriv = lambda x: self.MeSigmaIDeriv(x)
        MfMui = self.MfMui

        if adjoint:
            if self._makeASymmetric is True:
                v = MfMui * v
            return  MeSigmaIDeriv(C.T * ( MfMui * u )).T * ( C.T * v ) 

        ADeriv = ( C * ( MeSigmaIDeriv(C.T * ( MfMui * u )) * v ) )
        if self._makeASymmetric is True:
            return MfMui.T * ADeriv
        return ADeriv


    def getAsubdiag(self, tInd):

        dt = self.timeSteps[tInd]
        MfMui = self.MfMui
        Asubdiag = - 1./dt * sp.eye(self.mesh.nF)

        if self._makeASymmetric is True:
            return MfMui.T * Asubdiag

        return Asubdiag



    def getRHS(self, tInd):
        dt = self.timeSteps[tInd]
        C = self.mesh.edgeCurl
        MeSigmaI = self.MeSigmaI
        MfMui = self.MfMui

        S_m, S_e = self.getSourceTerm(tInd) 

        # B_n = np.c_[[F[src,'bSolution',tInd] for src in self.survey.srcList]]
        # if B_n.shape[0] is not 1:
        #     raise NotImplementedError('getRHS not implemented for this shape of B_n')

        rhs =  (C * (MeSigmaI * S_e) + S_m) # + 1./dt * B_n[:,:,0].T  
        if self._makeASymmetric is True:
            return MfMui.T * rhs
        return rhs

    def getRHSDeriv(self, tInd, src, v, adjoint=False):

        dt = self.timeSteps[tInd]
        C = self.mesh.edgeCurl
        MeSigmaI = self.MeSigmaI
        MeSigmaIDeriv = lambda u: self.MeSigmaIDeriv(u)
        MfMui = self.MfMui

        _, S_e = src.eval(tInd, self) # I think this is tInd+1 ? 
        S_mDeriv, S_eDeriv = src.evalDeriv(self.times[tInd+1], self, adjoint=adjoint) # I think this is tInd+1 ? 

        if adjoint:
            if self._makeASymmetric is True:
                v = self.MfMui * v
            if isinstance(S_e, Utils.Zero): 
                MeSigmaIDerivT_v = Utils.Zero()
            else: 
                MeSigmaIDerivT_v = MeSigmaIDeriv(S_e).T * v
            RHSDeriv = MeSigmaIDerivT_v + S_eDeriv( MeSigmaI.T *  ( C.T * v ) ) + S_mDeriv(v) #+ dbn_dm_v / dt #this will be given the transposed version
            return RHSDeriv

        if isinstance(S_e, Utils.Zero): 
            MeSigmaIDeriv_v = Utils.Zero()
        else: 
            MeSigmaIDeriv_v = MeSigmaIDeriv(S_e) * v

        RHSDeriv = (C * (MeSigmaIDeriv_v + MeSigmaI * S_eDeriv(v) + S_mDeriv(v))) #+ dbn_dm_v / dt  

        if self._makeASymmetric is True:
            return self.MfMui.T * RHSDeriv
        return RHSDeriv 


    @Utils.timeIt
    def getJdiags(self, tInd, adjoint = False):
        # The matrix that we are computing has the form:
        #
        #   -                                           -   -  -     -  -
        #  |  Adiag                                 | | uderiv1 |   | b1 |
        #  |   Asub    Adiag                        | | uderiv2 |   | b2 |
        #  |            Asub    Adiag               | | uderiv3 | = | b3 |
        #  |                 ...     ...            | |   ...   |   | .. |
        #  |                         Asub    Adiag  | | uderivn |   | bn |
        #   -                                           -   -  -     -  -
        
        if adjoint:
            raise NotImplementedError

        dt = self.timeSteps[tInd]

        Adiag = self.getA(tInd)
        Asub  = - 1./dt * Utils.speye(self.mesh.nF)

        if self._makeASymmetric:
            Asub = self.MfMui.T * Asub

        return Adiag, Asub





