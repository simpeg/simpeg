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

            rhs = self.getRHS(tInd+1) # this is on the nodes of the time mesh
            Asubdiag = self.getAsubdiag(tInd)

            if self.verbose: print ('    Solving...   (tInd = %i)')% (tInd+1)
            sol = Ainv * (rhs - Asubdiag * F[:,self._fieldType+'Solution',tInd]) # taking a step

            if self.verbose: print '    Done...'

            if sol.ndim == 1:
                sol.shape = (sol.size,1)
            F[:,self._fieldType+'Solution',tInd+1] = sol

        Ainv.clean()
        return F


    def Jvec(self, m, v, f=None):
        """
        Jvec computes the sensitivity times a vector

        .. math::
            \mathbf{J} \mathbf{v} = \\frac{d\mathbf{P}}{d\mathbf{F}} \left( \\frac{d\mathbf{F}}{d\mathbf{u}} \\frac{d\mathbf{u}}{d\mathbf{m}} + \\frac{\partial\mathbf{F}}{\partial\mathbf{m}} \\right) \mathbf{v}

        where

        .. math::
            \mathbf{A} \\frac{d\mathbf{u}}{d\mathbf{m}} + \\frac{d\mathbf{A}(\mathbf{u})}{d\mathbf{m}} = \\frac{d \mathbf{RHS}}{d \mathbf{m}}
        """

        if f is None:
           f = self.fields(m)

        ftype = self._fieldType + 'Solution' # the thing we solved for
        self.curModel = m

        Jv = self.dataPair(self.survey)

        # mat to store previous time-step's solution deriv times a vector for each source
        # size: nu x nSrc

        # this is a bit silly

        # if self._fieldType is 'b' or self._fieldType is 'j':
        #     ifields = np.zeros((self.mesh.nF, len(Srcs)))
        # elif self._fieldType is 'e' or self._fieldType is 'h':
        #     ifields = np.zeros((self.mesh.nE, len(Srcs)))

        # for i, src in enumerate(self.survey.srcList):
        dun_dm_v = np.hstack([Utils.mkvc(self.getInitialFieldsDeriv(src,v),2) for src in self.survey.srcList]) # can over-write this at each timestep
        #
        df_dm_v = Fields_Derivs(self.mesh, self.survey) # store the field derivs we need to project to calc full deriv

        Adiaginv = None

        for tInd, dt in zip(range(self.nT), self.timeSteps):
            if Adiaginv is not None and (tInd > 0 and dt != self.timeSteps[tInd - 1]):# keep factors if dt is the same as previous step b/c A will be the same
                Adiaginv.clean()
                Adiaginv = None

            if Adiaginv is None:
                A = self.getAdiag(tInd)
                Adiaginv = self.Solver(A, **self.solverOpts)

            Asubdiag = self.getAsubdiag(tInd)

            for i, src in enumerate(self.survey.srcList):

                # here, we are lagging by a timestep, so filling in as we go
                for projField in set([rx.projField for rx in src.rxList]):
                    df_dmFun = getattr(f, '_%sDeriv'%projField, None)
                    # df_dm_v is dense, but we only need the times at (rx.P.T * ones > 0)
                    # This should be called rx.footprint
                    df_dm_v[src, '%sDeriv'%projField , tInd] = df_dmFun(tInd, src, dun_dm_v[:,i], v)

                un_src = f[src,ftype,tInd+1]

                dA_dm_v   = self.getAdiagDeriv(tInd, un_src, v) # cell centered on time mesh
                dRHS_dm_v = self.getRHSDeriv(tInd+1, src, v) # on nodes of time mesh

                dAsubdiag_dm_v = self.getAsubdiagDeriv(tInd, f[src,ftype,tInd], v)

                JRHS = dRHS_dm_v - dAsubdiag_dm_v - dA_dm_v

                # step in time and overwrite
                if tInd != len(self.timeSteps+1):
                    dun_dm_v[:,i] = Adiaginv * (JRHS - Asubdiag * dun_dm_v[:,i])

        for src in self.survey.srcList:
            for rx in src.rxList:
                Jv[src,rx] = rx.evalDeriv(src, self.mesh, self.timeMesh, Utils.mkvc(df_dm_v[src,'%sDeriv'%rx.projField,:]))

        Adiaginv.clean()
        return Utils.mkvc(Jv)


    def Jtvec(self, m, v, f=None):

        """
        Jvec computes the adjoint of the sensitivity times a vector

        .. math::
            \mathbf{J}^\\top \mathbf{v} =  \left( \\frac{d\mathbf{u}}{d\mathbf{m}} ^ \\top \\frac{d\mathbf{F}}{d\mathbf{u}} ^ \\top  + \\frac{\partial\mathbf{F}}{\partial\mathbf{m}} ^ \\top \\right) \\frac{d\mathbf{P}}{d\mathbf{F}} ^ \\top \mathbf{v}

        where

        .. math::
            \\frac{d\mathbf{u}}{d\mathbf{m}} ^\\top \mathbf{A}^\\top  + \\frac{d\mathbf{A}(\mathbf{u})}{d\mathbf{m}} ^ \\top = \\frac{d \mathbf{RHS}}{d \mathbf{m}} ^ \\top
        """

        if f is None:
            f = self.fields(m)

        self.curModel = m
        ftype = self._fieldType + 'Solution' # the thing we solved for

        # Ensure v is a data object.
        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        df_duT_v = Fields_Derivs(self.mesh, self.survey)
        ATinv_df_duT_v = np.zeros((len(self.survey.srcList), len(f[self.survey.srcList[0],ftype,0]))) # same size as fields at a single timestep

        JTv = np.zeros(m.shape)

        # Loop over sources and receivers to create a fields object: PT_v, df_duT_v, df_dmT_v
        for src in self.survey.srcList:
            PT_v = Fields_Derivs(self.mesh, self.survey) # initialize storage for PT_v (don't need to preserve over sources)
            # initialize size
            df_duT_v[src, '%sDeriv'%self._fieldType, :] = np.zeros_like(f[src, self._fieldType, :])

            for rx in src.rxList:
                PT_v[src,'%sDeriv'%rx.projField,:] = rx.evalDeriv(src, self.mesh, self.timeMesh, Utils.mkvc(v[src,rx]), adjoint=True) # this is +=

                # PT_v = np.reshape(curPT_v,(len(curPT_v)/self.timeMesh.nN, self.timeMesh.nN), order='F')

                df_duTFun = getattr(f, '_%sDeriv'%rx.projField, None)

                for tInd in range(self.nT+1):
                    cur = df_duTFun(tInd, src, None, Utils.mkvc(PT_v[src,'%sDeriv'%rx.projField,tInd]), adjoint=True)

                    df_duT_v[src, '%sDeriv'%self._fieldType, tInd] = df_duT_v[src, '%sDeriv'%self._fieldType, tInd] + Utils.mkvc(cur[0],2)
                    JTv = cur[1] + JTv

        del PT_v # no longer need this


        AdiagTinv = None

        # Do the back-solve through time
        for tIndP in reversed(range(self.nT + 1)):
            tInd = tIndP - 1
            if AdiagTinv is not None and (tInd <= self.nT and self.timeSteps[tInd] != self.timeSteps[tInd+1]): # if the previous timestep is the same --> no need to refactor the matrix
                AdiagTinv.clean()
                AdiagTinv = None

            # refactor if we need to
            if AdiagTinv is None and tInd > -1:
                Adiag = self.getAdiag(tInd)
                AdiagTinv = self.Solver(Adiag.T, **self.solverOpts)

            dAsubdiag_dm_v = Zero()

            if tInd < self.nT - 1:
                Asubdiag = self.getAsubdiag(tInd+1)


            for isrc, src in enumerate(self.survey.srcList):
                # solve against df_duT_v
                if tInd >= self.nT-1:
                    # last timestep (first to be solved)
                    ATinv_df_duT_v[isrc,:] = AdiagTinv * df_duT_v[src,'%sDeriv'%self._fieldType,tInd+1]
                elif tInd > -1:
                # else:
                    ATinv_df_duT_v[isrc,:] = AdiagTinv * (Utils.mkvc(df_duT_v[src,'%sDeriv'%self._fieldType,tInd+1]) - Asubdiag.T * Utils.mkvc(ATinv_df_duT_v[isrc,:]))
                else:
                    # AdiagTinv = I
                    ATinv_df_duT_v[isrc,:] = Utils.mkvc(df_duT_v[src,'%sDeriv'%self._fieldType,tInd+1]) - Asubdiag.T * Utils.mkvc(ATinv_df_duT_v[isrc,:])
                    # - Utils.mkvc(Asubdiag.T * Utils.mkvc(ATinv_df_duT_v[isrc,:]))
                    # (Utils.mkvc(df_duT_v[src,'%sDeriv'%self._fieldType,tInd+1]) - Asubdiag.T * Utils.mkvc(ATinv_df_duT_v[isrc,:]))

                if tInd < self.nT - 1:
                    dAsubdiagT_dm_v = self.getAsubdiagDeriv(tInd+1, f[src,ftype,tInd+1], ATinv_df_duT_v[isrc,:], adjoint = True)

                if tInd > -1:
                    un_src = f[src,ftype,tInd+1]
                    dAT_dm_v = self.getAdiagDeriv(tInd, un_src, ATinv_df_duT_v[isrc,:], adjoint=True) # cell centered on time mesh
                    dRHST_dm_v = self.getRHSDeriv(tInd+1, src, ATinv_df_duT_v[isrc,:], adjoint=True) # on nodes of time mesh

                    JTv = JTv + Utils.mkvc(- dAT_dm_v - dAsubdiag_dm_v + dRHST_dm_v)
                else:
                    # dA_dm_v = self.getInitialFieldsDeriv(df_duT_v[src,'%sDeriv'%self._fieldType,tInd+1], adjoint=True)
                    # print np.linalg.norm(self.getInitialFieldsDeriv(src, df_duT_v[src,'%sDeriv'%self._fieldType,tInd+1], adjoint=True))
                    # print np.linalg.norm(df_duT_v[src,'%sDeriv'%self._fieldType,tInd+1])
                    # vec = - Asubdiag.T * Utils.mkvc(ATinv_df_duT_v[isrc,:]) + Utils.mkvc(df_duT_v[src,'%sDeriv'%self._fieldType,tInd+1])
                    # dAsubdiagT_dm_v = self.getAsubdiagDeriv(tInd+1, f[src,ftype,tInd+1], Utils.mkvc(ATinv_df_duT_v[isrc,:]), adjoint = True)
                    dRHST_dm_v = Utils.mkvc(self.getInitialFieldsDeriv(src, Utils.mkvc(ATinv_df_duT_v[isrc,:]) , adjoint=True))

                    JTv = JTv + Utils.mkvc( -dAsubdiagT_dm_v + dRHST_dm_v) #



                    # # dAT_dm_v = self.getAdiagDeriv(tInd, un_src, ATinv_df_duT_v[isrc,:], adjoint=True) # cell centered on time mesh
                    # dRHST_dm_v0 = self.getRHSDeriv(tInd+1, src, ATinv_df_duT_v[isrc,:], adjoint=True) # on nodes of time mesh
                    # dRHST_dm_v1 = self.getInitialFieldsDeriv( Utils.mkvc(df_duT_v[src,'%sDeriv'%self._fieldType,tInd+1]), adjoint=True)
                    # JTv = JTv + Utils.mkvc(dRHST_dm_v0 + dRHST_dm_v1)

                    # print 'here'
                    # inFields = self.getInitialFieldsDeriv(f[src,ftype,tInd+1], Utils.mkvc(df_duT_v[src,'%sDeriv'%self._fieldType,tInd+1]), adjoint=True)
                    #  # - Asubdiag.T * Utils.mkvc(ATinv_df_duT_v[isrc,:]), adjoint=True)
                    # print inFields.shape
                    # JTv = JTv + inFields
                # dAsubdiag_dm_v = 0



        # Missing the 0 step

        # adding du_dm^T * dF_du^T * P^T vfor time 0 (no dRHS_dm_v at time 0)
        # Asubdiag = self.getAsubdiag(0)
        # for src in self.survey.srcList:
        #     for projField in set(rx.projField):
        #         v = AdiagTinv * (Utils.mkvc(df_duT_v[src,'%sDeriv'%self._fieldType,0]) - Asubdiag.T * Utils.mkvc(ATinv_df_duT_v[isrc,:]))
        #         JTv = JTv - Utils.mkvc(self.getAdiagDeriv(0, f[src, ftype, tInd], v, adjoint = True))
        #         # JTv = JTv + self.getInitialFieldsDeriv(Utils.mkvc(df_duT_v[src,'%sDeriv'%self._fieldType,0] - Asubdiag.T * Utils.mkvc(ATinv_df_duT_v[isrc,:])), adjoint=True)


        return Utils.mkvc(JTv).astype(float)



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

    def getInitialFieldsDeriv(self, src, v, adjoint=False):

        if adjoint is False:
            if self._fieldType is 'b' or self._fieldType is 'j':
                ifieldsDeriv = np.zeros(self.mesh.nF)
            elif self._fieldType is 'e' or self._fieldType is 'h':
                ifieldsDeriv = np.zeros(self.mesh.nE)

        elif adjoint is True:
            ifieldsDeriv = np.zeros(self.mapping.nP)

        ifieldsDeriv = Utils.mkvc(getattr(src, '%sInitialDeriv'%self._fieldType, None)(self,v,adjoint)) + ifieldsDeriv

            # ifieldsDeriv = Utils.mkvc(getattr(src, '%sInitialDeriv'%self._fieldType, None)(self,v,adjoint)) + ifieldsDeriv
            # ifieldsDeriv = self.getAdiagDeriv(None, u, v, adjoint)
            # ifieldsDeriv = ifieldsDeriv.sum()

        return ifieldsDeriv


##########################################################################################
################################ E-B Formulation #########################################
##########################################################################################

# ------------------------------- Problem_b -------------------------------------------- #

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
        assert tInd >= 0 and tInd < self.nT

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

    def getAsubdiagDeriv(self, tInd, u, v, adjoint=False):
        return Zero() * v



    def getRHS(self, tInd):
        C = self.mesh.edgeCurl
        MeSigmaI = self.MeSigmaI
        MfMui = self.MfMui

        S_m, S_e = self.getSourceTerm(tInd)

        rhs =  (C * (MeSigmaI * S_e) + S_m)
        if self._makeASymmetric is True:
            return MfMui.T * rhs
        return rhs

    def getRHSDeriv(self, tInd, src, v, adjoint=False):

        C = self.mesh.edgeCurl
        MeSigmaI = self.MeSigmaI
        MeSigmaIDeriv = lambda u: self.MeSigmaIDeriv(u)
        MfMui = self.MfMui

        _, S_e = src.eval(tInd, self)
        S_mDeriv, S_eDeriv = src.evalDeriv(self.times[tInd], self, adjoint=adjoint)

        if adjoint:
            if self._makeASymmetric is True:
                v = self.MfMui * v
            if isinstance(S_e, Utils.Zero):
                MeSigmaIDerivT_v = Utils.Zero()
            else:
                MeSigmaIDerivT_v = MeSigmaIDeriv(S_e).T * v
            RHSDeriv = MeSigmaIDerivT_v + S_eDeriv( MeSigmaI.T *  ( C.T * v ) ) + S_mDeriv(v)
            return RHSDeriv

        if isinstance(S_e, Utils.Zero):
            MeSigmaIDeriv_v = Utils.Zero()
        else:
            MeSigmaIDeriv_v = MeSigmaIDeriv(S_e) * v

        RHSDeriv = (C * (MeSigmaIDeriv_v + MeSigmaI * S_eDeriv(v) + S_mDeriv(v)))

        if self._makeASymmetric is True:
            return self.MfMui.T * RHSDeriv
        return RHSDeriv


# ------------------------------- Problem_e -------------------------------------------- #

class Problem_e(BaseTDEMProblem):

    _fieldType = 'e'
    _eqLocs    = 'FE'
    fieldsPair = Fields_e
    surveyPair = SurveyTDEM

    def __init__(self, mesh, mapping=None, **kwargs):
        BaseTDEMProblem.__init__(self, mesh, mapping=mapping, **kwargs)

    def getAdiag(self, tInd):
        """
        System matrix at a given time index

        """
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]
        C = self.mesh.edgeCurl
        MfMui = self.MfMui
        MeSigma = self.MeSigma

        return C.T * ( MfMui * C ) + 1./dt * MeSigma


    def getAdiagDeriv(self, tInd, u, v, adjoint=False):
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]
        C = self.mesh.edgeCurl
        MfMui = self.MfMui
        MeSigmaDeriv = self.MeSigmaDeriv(u)

        if adjoint:
            return 1./dt * MeSigmaDeriv.T * v

        return 1./dt * MeSigmaDeriv * v


    def getAsubdiag(self, tInd):
        assert tInd >= 0 and tInd < self.nT

        dt = self.timeSteps[tInd]

        return - 1./dt * self.MeSigma

    def getAsubdiagDeriv(self, tInd, u, v, adjoint=False):
        dt = self.timeSteps[tInd]

        if adjoint:
            return - 1./dt * self.MeSigmaDeriv(u).T * v

        return - 1./dt * self.MeSigmaDeriv(u) * v

    def getRHS(self, tInd):
        return Zero()

    def getRHSDeriv(self, tInd, src, v, adjoint=False):
        return Zero()





