import Utils, numpy as np


class Parameter(object):
    """Parameter"""

    debug    = False    #: Print debugging information

    current = None     #: This hold
    currentIter = 0

    def __init__(self, **kwargs):
        Utils.setKwargs(self, **kwargs)

    @property
    def parent(self):
        """This is the parent of the Parameter instance."""
        return getattr(self,'_parent',None)
    @parent.setter
    def parent(self, p):
        startupName = '_startup_paramProperty_'+self._propertyName
        if getattr(self,'_parent',None) is not None:
            delattr(self._parent,startupName)
            print 'Warning: Parameter %s has switched to a new parent.' % self._propertyName
            if self.debug: print '%s function has been deleted' % startupName
        self._parent = p

        prop = self
        def _startup_paramProperty(self, *args):
            if prop.debug: print 'initializing %s' % prop._propertyName
            prop.initialize()

        Utils.hook(self._parent, _startup_paramProperty, name=startupName, overwrite=True)

    @property
    def inv(self): return self.parent.inv
    @property
    def objFunc(self): return self.parent.objFunc
    @property
    def opt(self): return self.parent.opt
    @property
    def reg(self): return self.parent.reg
    @property
    def data(self): return self.parent.data
    @property
    def prob(self): return self.parent.prob
    @property
    def model(self): return self.parent.model
    @property
    def mesh(self): return self.parent.mesh

    def initialize(self):
        pass

    def get(self):
        if (self.current is None or
            not self.opt.iter == self.currentIter):
            self.current = self.nextIter()
            self.currentIter = self.opt.iter
        return self.current

    def nextIter(self):
        raise NotImplementedError('Getting the Parameter is not yet implemented.')


def ParameterProperty(name, default=None, doc=""):
    def getter(self):
        out = getattr(self,'_'+name,default)
        if isinstance(out, Parameter):
            out = out.get()
        return out
    def setter(self, value):
        if isinstance(value, Parameter):
            value._propertyName = name
            value.parent = self
        setattr(self, '_'+name, value)

    return property(fget=getter, fset=setter, doc=doc)


class BetaEstimate(Parameter):
    """BetaEstimate"""

    beta0 = 'guess'      #: The initial Beta (regularization parameter)
    beta0_ratio = 0.1    #: When beta0 is set to 'guess', estimateBeta0 is used with this ratio

    beta = None          #: Beta parameter

    def __init__(self, **kwargs):
        Parameter.__init__(self, **kwargs)

    def initialize(self):
        self.beta = self.beta0

    @Utils.requires('parent')
    def nextIter(self):
        if self.beta is 'guess':
            if self.debug: print 'BetaSchedule is estimating Beta0.'
            self.beta = self.estimateBeta0()
        return self.beta

    @Utils.requires('parent')
    def estimateBeta0(self):
        """estimateBeta0(u=None)

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
        objFunc  = self.parent
        data     = objFunc.data

        m = objFunc.m_current
        u = objFunc.u_current

        if u is None:
            u = data.prob.fields(m)

        x0 = np.random.rand(*m.shape)
        t = x0.dot(objFunc.dataObj2Deriv(m,x0,u=u))
        b = x0.dot(objFunc.reg.modelObj2Deriv(m, v=x0))
        return self.beta0_ratio*(t/b)


class BetaSchedule(BetaEstimate):
    """BetaSchedule"""

    coolingFactor = 2.
    coolingRate = 3

    @Utils.requires('parent')
    def nextIter(self):
        if self.beta is 'guess':
            if self.debug: print 'BetaSchedule is estimating Beta0.'
            self.beta = self.estimateBeta0()

        if self.opt.iter > 0 and self.opt.iter % self.coolingRate == 0:
            if self.debug: print 'BetaSchedule is cooling Beta. Iteration: %d' % self.opt.iter
            self.beta /= self.coolingFactor

        return self.beta
