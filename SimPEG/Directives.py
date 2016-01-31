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
        u = self.invProb.getFields(m, store=True, deleteWarmstart=False)

        x0 = np.random.rand(*m.shape)
        t = x0.dot(self.dmisfit.eval2Deriv(m,x0,u=u))
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

    @property
    def target(self):
        if getattr(self, '_target', None) is None:
            self._target = self.survey.nD*0.5
        return self._target
    @target.setter
    def target(self, val):
        self._target = val

    def endIter(self):
        if self.invProb.phi_d < self.target:
            self.opt.stopNextIteration = True



class _SaveEveryIteration(InversionDirective):
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


class SaveModelEveryIteration(_SaveEveryIteration):
    """SaveModelEveryIteration"""

    def initialize(self):
        print "SimPEG.SaveModelEveryIteration will save your models as: '###-%s.npy'"%self.fileName

    def endIter(self):
        np.save('%03d-%s' % (self.opt.iter, self.fileName), self.opt.xc)


class SaveOutputEveryIteration(_SaveEveryIteration):
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

class SaveOutputDictEveryIteration(_SaveEveryIteration):
    """SaveOutputDictEveryIteration"""

    def initialize(self):
        print "SimPEG.SaveOutputDictEveryIteration will save your inversion progress as dictionary: '###-%s.npz'"%self.fileName

    def endIter(self):
        # Save the data.
        ms = self.reg.Ws * ( self.reg.mapping * (self.invProb.curModel - self.reg.mref) )
        phi_ms = 0.5*ms.dot(ms)
        if self.reg.smoothModel == True:
            mref = self.reg.mref
        else:
            mref = 0
        mx = self.reg.Wx * ( self.reg.mapping * (self.invProb.curModel - mref) )
        phi_mx = 0.5 * mx.dot(mx)
        if self.prob.mesh.dim==2:
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



# class UpdateReferenceModel(Parameter):

#     mref0 = None

#     def nextIter(self):
#         mref = getattr(self, 'm_prev', None)
#         if mref is None:
#             if self.debug: print 'UpdateReferenceModel is using mref0'
#             mref = self.mref0
#         self.m_prev = self.invProb.m_current
#         return mref
