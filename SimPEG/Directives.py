import Utils, numpy as np

class InversionDirective(object):
    """InversionDirective"""

    debug = False    #: Print debugging information

    def __init__(self, **kwargs):
        Utils.setKwargs(self, **kwargs)

    @property
    def inversion(self):
        """This is the inversion of the InversionDirective instance."""
        return getattr(self,'_inversion',None)
    @inversion.setter
    def inversion(self, i):
        if getattr(self,'_inversion',None) is not None:
            print 'Warning: InversionDirective %s has switched to a new inversion.' % self.__name__
        self._inversion = i

    @property
    def invProb(self): return self.inversion.invProb
    @property
    def opt(self): return self.invProb.opt
    @property
    def reg(self): return self.invProb.reg
    @property
    def dmisfit(self): return self.invProb.dmisfit
    @property
    def survey(self): return self.dmisfit.survey
    @property
    def prob(self): return self.dmisfit.prob

    def initialize(self):
        pass

    def endIter(self):
        pass

    def finish(self):
        pass

class DirectiveList(object):

    dList = None   #: The list of Directives

    def __init__(self, *directives, **kwargs):
        self.dList = []
        for d in directives:
            assert isinstance(d, InversionDirective), 'All directives must be InversionDirectives not %s' % d.__name__
            self.dList.append(d)
        Utils.setKwargs(self, **kwargs)

    @property
    def debug(self):
        return getattr(self, '_debug', False)
    @debug.setter
    def debug(self, value):
        for d in self.dList:
            d.debug = value
        self._debug = value

    @property
    def inversion(self):
        """This is the inversion of the InversionDirective instance."""
        return getattr(self,'_inversion',None)
    @inversion.setter
    def inversion(self, i):
        if self.inversion is i: return
        if getattr(self,'_inversion',None) is not None:
            print 'Warning: %s has switched to a new inversion.' % self.__name__
        for d in self.dList:
            d.inversion = i
        self._inversion = i

    def call(self, ruleType):
        if self.dList is None:
            if self.debug: 'DirectiveList is None, no directives to call!'
            return

        directives = ['initialize', 'endIter', 'finish']
        assert ruleType in directives, 'Directive type must be in ["%s"]' % '", "'.join(directives)
        for r in self.dList:
            getattr(r, ruleType)()


class BetaEstimate_ByEig(InversionDirective):
    """BetaEstimate"""

    beta0 = None       #: The initial Beta (regularization parameter)
    beta0_ratio = 1e2  #: estimateBeta0 is used with this ratio

    def initialize(self):
        """
            The initial beta is calculated by comparing the estimated
            eigenvalues of JtJ and WtW.

            To estimate the eigenvector of **A**, we will use one iteration
            of the *Power Method*:

            .. math::

                \mathbf{x_1 = A x_0}

            Given this (very course) approximation of the eigenvector,
            we can use the *Rayleigh quotient* to approximate the largest eigenvalue.

            .. math::

                \lambda_0 = \\frac{\mathbf{x^\\top A x}}{\mathbf{x^\\top x}}

            We will approximate the largest eigenvalue for both JtJ and WtW, and
            use some ratio of the quotient to estimate beta0.

            .. math::

                \\beta_0 = \gamma \\frac{\mathbf{x^\\top J^\\top J x}}{\mathbf{x^\\top W^\\top W x}}

            :rtype: float
            :return: beta0
        """

        if self.debug: print 'Calculating the beta0 parameter.'

        m = self.invProb.curModel
        f = self.invProb.getFields(m, store=True, deleteWarmstart=False)

        x0 = np.random.rand(*m.shape)
        t = x0.dot(self.dmisfit.eval2Deriv(m,x0,f=f))
        b = x0.dot(self.reg.eval2Deriv(m, v=x0))
        self.beta0 = self.beta0_ratio*(t/b)

        self.invProb.beta = self.beta0


class BetaSchedule(InversionDirective):
    """BetaSchedule"""

    coolingFactor = 8.
    coolingRate = 3

    def endIter(self):
        if self.opt.iter > 0 and self.opt.iter % self.coolingRate == 0:
            if self.debug: print 'BetaSchedule is cooling Beta. Iteration: %d' % self.opt.iter
            self.invProb.beta /= self.coolingFactor


class TargetMisfit(InversionDirective):

    chifact = 1.
    phi_d_star = None

    @property
    def target(self):
        if getattr(self, '_target', None) is None:
            if self.phi_d_star is None:
                self.phi_d_star = 0.5 * self.survey.nD
            self._target = self.chifact * self.phi_d_star # the factor of 0.5 is because we do phid = 0.5*|| dpred - dobs||^2
        return self._target
    @target.setter
    def target(self, val):
        self._target = val

    def endIter(self):
        if self.invProb.phi_d < self.target:
            self.opt.stopNextIteration = True



class SaveEveryIteration(InversionDirective):
    @property
    def name(self):
        if getattr(self, '_name', None) is None:
            self._name = 'InversionModel'
        return self._name
    @name.setter
    def name(self, value):
        self._name = value

    @property
    def fileName(self):
        if getattr(self, '_fileName', None) is None:
            from datetime import datetime
            self._fileName = '%s-%s'%(self.name, datetime.now().strftime('%Y-%m-%d-%H-%M'))
        return self._fileName
    @fileName.setter
    def fileName(self, value):
        self._fileName = value


class SaveModelEveryIteration(SaveEveryIteration):
    """SaveModelEveryIteration"""

    def initialize(self):
        print "SimPEG.SaveModelEveryIteration will save your models as: '###-%s.npy'"%self.fileName

    def endIter(self):
        np.save('%03d-%s' % (self.opt.iter, self.fileName), self.opt.xc)


class SaveOutputEveryIteration(SaveEveryIteration):
    """SaveModelEveryIteration"""

    def initialize(self):
        print "SimPEG.SaveOutputEveryIteration will save your inversion progress as: '###-%s.txt'"%self.fileName
        f = open(self.fileName+'.txt', 'w')
        f.write("  #     beta     phi_d     phi_m       f\n")
        f.close()

    def endIter(self):
        f = open(self.fileName+'.txt', 'a')
        f.write(' %3d %1.4e %1.4e %1.4e %1.4e\n'%(self.opt.iter, self.invProb.beta, self.invProb.phi_d, self.invProb.phi_m, self.opt.f))
        f.close()

class SaveOutputDictEveryIteration(SaveEveryIteration):
    """SaveOutputDictEveryIteration"""

    def initialize(self):
        print "SimPEG.SaveOutputDictEveryIteration will save your inversion progress as dictionary: '###-%s.npz'"%self.fileName

    def endIter(self):
        # Save the data.
        ms = self.reg.Ws * ( self.reg.mapping * (self.invProb.curModel - self.reg.mref) )
        phi_ms = 0.5*ms.dot(ms)
        if self.reg.mrefInSmooth == True:
            mref = self.reg.mref
        else:
            mref = 0
        mx = self.reg.Wx * ( self.reg.mapping * (self.invProb.curModel - mref) )
        phi_mx = 0.5 * mx.dot(mx)
        if self.prob.mesh.dim >= 2:
            my = self.reg.Wy * ( self.reg.mapping * (self.invProb.curModel - mref) )
            phi_my = 0.5 * my.dot(my)
        else:
            phi_my = 'NaN'
        if self.prob.mesh.dim==3:
            mz = self.reg.Wz * ( self.reg.mapping * (self.invProb.curModel - mref) )
            phi_mz = 0.5 * mz.dot(mz)
        else:
            phi_mz = 'NaN'


        # Save the file as a npz
        np.savez('{:03d}-{:s}'.format(self.opt.iter,self.fileName), iter=self.opt.iter, beta=self.invProb.beta, phi_d=self.invProb.phi_d, phi_m=self.invProb.phi_m, phi_ms=phi_ms, phi_mx=phi_mx, phi_my=phi_my, phi_mz=phi_mz,f=self.opt.f, m=self.invProb.curModel,dpred=self.invProb.dpred)

#         mref = getattr(self, 'm_prev', None)
#         if mref is None:
#             if self.debug: print 'UpdateReferenceModel is using mref0'
#             mref = self.mref0
#         self.m_prev = self.invProb.m_current
#         return mref

class Update_IRLS(InversionDirective):

    eps_min = None
    eps_p = None
    eps_q = None
    norms = [2.,2.,2.,2.]
    factor = None
    gamma = None
    phi_m_last = None
    phi_d_last = None
    f_old = None
    f_min_change = 1e-2
    beta_tol = 5e-2

    # Solving parameter for IRLS (mode:2)
    IRLSiter   = 0
    minGNiter = 5
    maxIRLSiter = 10
    iterStart = 0

    # Beta schedule
    coolingFactor = 2.
    coolingRate = 1

    mode = 1

    @property
    def target(self):
        if getattr(self, '_target', None) is None:
            self._target = self.survey.nD*0.5
        return self._target
    @target.setter
    def target(self, val):
        self._target = val

    def initialize(self):

        if self.mode == 1:
            self.reg.norms = [2., 2., 2., 2.]

    def endIter(self):

        # After reaching target misfit with l2-norm, switch to IRLS (mode:2)
        if self.invProb.phi_d < self.target and self.mode == 1:
            print "Convergence with smooth l2-norm regularization: Start IRLS steps..."

            self.mode = 2
            print self.eps_p, self.eps_q, self.norms
            self.reg.eps_p = self.eps_p
            self.reg.eps_q = self.eps_q
            self.reg.norms = self.norms
            self.coolingFactor = 1.
            self.coolingRate = 1
            self.iterStart = self.opt.iter
            self.phi_d_last = self.invProb.phi_d
            self.phi_m_last = self.invProb.phi_m_last

            self.reg.l2model = self.invProb.curModel
            self.reg.curModel = self.invProb.curModel

            if getattr(self, 'f_old', None) is None:
                self.f_old = self.reg.eval(self.invProb.curModel)#self.invProb.evalFunction(self.invProb.curModel, return_g=False, return_H=False)

        # Beta Schedule
        if self.opt.iter > 0 and self.opt.iter % self.coolingRate == 0:
            if self.debug: print 'BetaSchedule is cooling Beta. Iteration: %d' % self.opt.iter
            self.invProb.beta /= self.coolingFactor


        # Only update after GN iterations
        if (self.opt.iter-self.iterStart) % self.minGNiter == 0 and self.mode==2:

            self.IRLSiter += 1

            phim_new = self.reg.eval(self.invProb.curModel)
            self.f_change = np.abs(self.f_old - phim_new) / self.f_old

            print "Regularization decrease: %6.3e" % (self.f_change)

            # Check for maximum number of IRLS cycles
            if self.IRLSiter == self.maxIRLSiter:
                print "Reach maximum number of IRLS cycles: %i" % self.maxIRLSiter
                self.opt.stopNextIteration = True
                return

            # Check if the function has changed enough
            if self.f_change < self.f_min_change and self.IRLSiter > 1:
                print "Minimum decrease in regularization. End of IRLS"
                self.opt.stopNextIteration = True
                return
            else:
                self.f_old = phim_new

            # Cool the threshold parameter if required
            if getattr(self, 'factor', None) is not None:
                eps = self.reg.eps / self.factor

                if getattr(self, 'eps_min', None) is not None:
                    self.reg.eps = np.max([self.eps_min,eps])
                else:
                    self.reg.eps = eps

            # Get phi_m at the end of current iteration
            self.phi_m_last = self.invProb.phi_m_last

            # Reset the regularization matrices so that it is
            # recalculated for current model
            self.reg._Wsmall = None
            self.reg._Wx = None
            self.reg._Wy = None
            self.reg._Wz = None

             # Update the model used for the IRLS weights
            self.reg.curModel = self.invProb.curModel

            # Temporarely set gamma to 1. to get raw phi_m
            self.reg.gamma = 1.

            # Compute new model objective function value
            phim_new = self.reg.eval(self.invProb.curModel)

            # Update gamma to scale the regularization between IRLS iterations
            self.reg.gamma = self.phi_m_last / phim_new

            # Reset the regularization matrices again for new gamma
            self.reg._Wsmall = None
            self.reg._Wx = None
            self.reg._Wy = None
            self.reg._Wz = None

            # Check if misfit is within the tolerance, otherwise scale beta
            val = self.invProb.phi_d / (self.survey.nD*0.5)

            if np.abs(1.-val) > self.beta_tol:
                self.invProb.beta = self.invProb.beta * self.survey.nD*0.5 / self.invProb.phi_d

class Update_lin_PreCond(InversionDirective):
    """
    Create a Jacobi preconditioner for the linear problem
    """
    onlyOnStart=False

    def initialize(self):

        if getattr(self.opt, 'approxHinv', None) is None:
            # Update the pre-conditioner
            diagA = np.sum(self.prob.G**2.,axis=0) + self.invProb.beta*(self.reg.W.T*self.reg.W).diagonal() #* (self.reg.mapping * np.ones(self.reg.curModel.size))**2.
            PC     = Utils.sdiag((self.prob.mapping.deriv(None).T *diagA)**-1.)
            self.opt.approxHinv = PC

    def endIter(self):
        # Cool the threshold parameter
        if self.onlyOnStart==True:
            return

        if getattr(self.opt, 'approxHinv', None) is not None:
            # Update the pre-conditioner
            diagA = np.sum(self.prob.G**2.,axis=0) + self.invProb.beta*(self.reg.W.T*self.reg.W).diagonal() #* (self.reg.mapping * np.ones(self.reg.curModel.size))**2.
            PC     = Utils.sdiag((self.prob.mapping.deriv(None).T *diagA)**-1.)
            self.opt.approxHinv = PC


class Update_Wj(InversionDirective):
    """
        Create approx-sensitivity base weighting using the probing method
    """
    k = None # Number of probing cycles
    itr = None # Iteration number to update Wj, or always update if None

    def endIter(self):

        if self.itr is None or self.itr == self.opt.iter:

            m = self.invProb.curModel
            if self.k is None:
                self.k = int(self.survey.nD/10)

            def JtJv(v):

                Jv = self.prob.Jvec(m, v)

                return self.prob.Jtvec(m,Jv)

            JtJdiag = Utils.diagEst(JtJv,len(m),k=self.k)
            JtJdiag = JtJdiag / max(JtJdiag)

            self.reg.wght = JtJdiag
