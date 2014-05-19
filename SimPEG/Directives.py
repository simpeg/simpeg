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
    beta0_ratio = 0.1  #: estimateBeta0 is used with this ratio

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

    coolingFactor = 2.
    coolingRate = 3

    def endIter(self):
        if self.opt.iter > 0 and self.opt.iter % self.coolingRate == 0:
            if self.debug: print 'BetaSchedule is cooling Beta. Iteration: %d' % self.opt.iter
            self.invProb.beta /= self.coolingFactor


# class UpdateReferenceModel(Parameter):

#     mref0 = None

#     def nextIter(self):
#         mref = getattr(self, 'm_prev', None)
#         if mref is None:
#             if self.debug: print 'UpdateReferenceModel is using mref0'
#             mref = self.mref0
#         self.m_prev = self.invProb.m_current
#         return mref
