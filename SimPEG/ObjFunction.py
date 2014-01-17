from SimPEG import Utils, np, sp

class BaseObjFunction(object):
    """docstring for BaseObjFunction"""

    __metaclass__ = Utils.Save.Savable

    beta    = None   #: Regularization trade-off parameter
    debug   = False  #: Print debugging information
    counter = None   #: Set this to a SimPEG.Utils.Counter() if you want to count things

    name = 'BaseObjFunction'   #: Name of the objective function

    u_current = None   #: The most current evaluated field
    m_current = None   #: The most current model

    @property
    def parent(self):
        """This is the parent of the objective function."""
        return getattr(self,'_parent',None)
    @parent.setter
    def parent(self, p):
        if getattr(self,'_parent',None) is not None:
            print 'Objective function has switched to a new parent!'
        self._parent = p


    def __init__(self, data, reg, **kwargs):
        Utils.setKwargs(self, **kwargs)

        self.data = data
        self.reg = reg


    @Utils.callHooks('startup')
    def startup(self, m0):
        """startup(m0)

            Called when inversion is first starting.
        """
        if self.debug: print 'Calling ObjFunction.startup'

        if not hasattr(self.reg, '_mref'):
            print 'Regularization has not set mref. SimPEG will set it to m0.'
            self.reg.mref = m0

        self.phi_d = np.nan
        self.phi_m = np.nan

        self.m_current = m0

    @Utils.timeIt
    def evalFunction(self, m, return_g=True, return_H=True):
        """evalFunction(m, return_g=True, return_H=True)
        """

        self.u_current = None
        self.m_current = m

        u = self.data.prob.field(m)
        self.u_current = u

        phi_d = self.dataObj(m, u=u)
        phi_m = self.reg.modelObj(m)

        self.dpred = self.data.dpred(m, u=u)  # This is a cheap matrix vector calculation.

        self.phi_d, self.phi_d_last  = phi_d, self.phi_d
        self.phi_m, self.phi_m_last  = phi_m, self.phi_m

        self._beta = self.beta.get()  #TODO: This needs to be fixed.

        f = phi_d + self._beta * phi_m

        out = (f,)
        if return_g:
            phi_dDeriv = self.dataObjDeriv(m, u=u)
            phi_mDeriv = self.reg.modelObjDeriv(m)

            g = phi_dDeriv + self._beta * phi_mDeriv
            out += (g,)

        if return_H:
            def H_fun(v):
                phi_d2Deriv = self.dataObj2Deriv(m, v, u=u)
                phi_m2Deriv = self.reg.modelObj2Deriv()*v

                return phi_d2Deriv + self._beta * phi_m2Deriv

            operator = sp.linalg.LinearOperator( (m.size, m.size), H_fun, dtype=m.dtype )
            out += (operator,)
        return out if len(out) > 1 else out[0]

    @Utils.timeIt
    def dataObj(self, m, u=None):
        """dataObj(m, u=None)

            :param numpy.array m: geophysical model
            :param numpy.array u: fields
            :rtype: float
            :return: data misfit

            The data misfit using an l_2 norm is:

            .. math::

                \mu_\\text{data} = {1\over 2}\left| \mathbf{W} \circ (\mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs}) \\right|_2^2

            Where P is a projection matrix that brings the field on the full domain to the data measurement locations;
            u is the field of interest; d_obs is the observed data; and W is the weighting matrix.
        """
        # TODO: ensure that this is a data is vector and Wd is a matrix.
        R = self.data.residualWeighted(m, u=u)
        return 0.5*np.vdot(R, R)

    @Utils.timeIt
    def dataObjDeriv(self, m, u=None):
        """dataObjDeriv(m, u=None)

            :param numpy.array m: geophysical model
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: data misfit derivative

            The data misfit using an l_2 norm is:

            .. math::

                \mu_\\text{data} = {1\over 2}\left| \mathbf{W} \circ (\mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs}) \\right|_2^2

            If the field, u, is provided, the calculation of the data is fast:

            .. math::

                \mathbf{d}_\\text{pred} = \mathbf{Pu(m)}

                \mathbf{R} = \mathbf{W} \circ (\mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs})

            Where P is a projection matrix that brings the field on the full domain to the data measurement locations;
            u is the field of interest; d_obs is the observed data; and W is the weighting matrix.

            The derivative of this, with respect to the model, is:

            .. math::

                \\frac{\partial \mu_\\text{data}}{\partial \mathbf{m}} = \mathbf{J}^\\top \mathbf{W \circ R}

        """
        if u is None: u = self.data.prob.field(m)

        R = self.data.residualWeighted(m, u=u)

        dmisfit = self.data.prob.Jt(m, self.data.Wd * R, u=u)

        return dmisfit

    @Utils.timeIt
    def dataObj2Deriv(self, m, v, u=None):
        """dataObj2Deriv(m, v, u=None)

            :param numpy.array m: geophysical model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: data misfit derivative

            The data misfit using an l_2 norm is:

            .. math::

                \mu_\\text{data} = {1\over 2}\left| \mathbf{W} \circ (\mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs}) \\right|_2^2

            If the field, u, is provided, the calculation of the data is fast:

            .. math::

                \mathbf{d}_\\text{pred} = \mathbf{Pu(m)}

                \mathbf{R} = \mathbf{W} \circ (\mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs})

            Where P is a projection matrix that brings the field on the full domain to the data measurement locations;
            u is the field of interest; d_obs is the observed data; and W is the weighting matrix.

            The derivative of this, with respect to the model, is:

            .. math::

                \\frac{\partial \mu_\\text{data}}{\partial \mathbf{m}} = \mathbf{J}^\\top \mathbf{W \circ R}

                \\frac{\partial^2 \mu_\\text{data}}{\partial^2 \mathbf{m}} = \mathbf{J}^\\top \mathbf{W \circ W J}

        """
        if u is None: u = self.data.prob.field(m)

        R = self.data.residualWeighted(m, u=u)

        # TODO: abstract to different norms a little cleaner.
        #                                                 \/ it goes here. in l2 it is the identity.
        dmisfit = self.data.prob.Jt_approx(m, self.data.Wd * self.data.Wd * self.data.prob.J_approx(m, v, u=u), u=u)

        return dmisfit



class BetaSchedule(Utils.Parameter):
    """BetaSchedule"""

    beta0 = 'guess'      #: The initial Beta (regularization parameter)
    beta0_ratio = 0.1    #: When beta0 is set to 'guess', estimateBeta0 is used with this ratio

    coolingFactor = 2.
    coolingRate = 3

    beta = None          #: Beta parameter

    def __init__(self, *args, **kwargs):
        Utils.Parameter.__init__(self, *args, **kwargs)
        Utils.setKwargs(self, **kwargs)

    def initialize(self):
        self.beta = self.beta0

    @Utils.requires('parent')
    def nextIter(self):
        if self.beta is 'guess':
            if self.debug: print 'BetaSchedule is estimating Beta0.'
            self.beta = self.estimateBeta0()

        opt = self.parent.parent.opt
        if opt._iter > 0 and opt._iter % self.coolingRate == 0:
            if self.debug: print 'BetaSchedule is cooling Beta. Iteration: %d' % opt._iter
            self.beta /= self.coolingFactor

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


            :param numpy.array u: fields
            :param float ratio: desired ratio of the eigenvalues, default is 0.1
            :rtype: float
            :return: beta0
        """
        objFunc  = self.parent
        data     = objFunc.data

        m = objFunc.m_current
        u = objFunc.u_current

        if u is None:
            u = data.prob.field(m)

        x0 = np.random.rand(*m.shape)
        t = x0.dot(objFunc.dataObj2Deriv(m,x0,u=u))
        b = x0.dot(objFunc.reg.modelObj2Deriv()*x0)
        return self.beta0_ratio*(t/b)
