import SimPEG
from SimPEG import utils, sp, np
from Optimize import Remember
from BetaSchedule import Cooling
from SimPEG.inverse import IterationPrinters, StoppingCriteria

class BaseInversion(object):
    """docstring for BaseInversion"""

    __metaclass__ = utils.Save.Savable

    maxIter = 1        #: Maximum number of iterations
    name = 'BaseInversion'

    debug   = False    #: Print debugging information

    comment = ''       #: Used by some functions to indicate what is going on in the algorithm
    counter = None     #: Set this to a SimPEG.utils.Counter() if you want to count things

    beta0  = None      #: The initial Beta (regularization parameter)
    beta0_ratio = 0.1  #: When beta0 is set to None, estimateBeta0 is used with this ratio

    def __init__(self, prob, reg, opt, **kwargs):
        utils.setKwargs(self, **kwargs)
        self.prob = prob
        self.reg = reg
        self.opt = opt
        self.opt.parent = self

        self.stoppers = [StoppingCriteria.iteration]

        # Check if we have inserted printers into the optimization
        if IterationPrinters.phi_d not in self.opt.printers:
            self.opt.printers.insert(1,IterationPrinters.beta)
            self.opt.printers.insert(2,IterationPrinters.phi_d)
            self.opt.printers.insert(3,IterationPrinters.phi_m)

        if not hasattr(opt, '_bfgsH0') and hasattr(opt, 'bfgsH0'): # Check if it has been set by the user and the default is not being used.
            print 'Setting bfgsH0 to the inverse of the modelObj2Deriv. Done using direct methods.'
            opt.bfgsH0 = SimPEG.Solver(reg.modelObj2Deriv())


    @property
    def Wd(self):
        """
            Standard deviation weighting matrix.
        """
        if getattr(self,'_Wd',None) is None:
            eps = np.linalg.norm(utils.mkvc(self.prob.dobs),2)*1e-5
            self._Wd = 1/(abs(self.prob.dobs)*self.prob.std+eps)
        return self._Wd
    @Wd.setter
    def Wd(self, value):
        self._Wd = value

    @property
    def phi_d_target(self):
        """
        target for phi_d

        By default this is the number of data.

        Note that we do not set the target if it is None, but we return the default value.
        """
        if getattr(self, '_phi_d_target', None) is None:
            return self.prob.dobs.size #
        return self._phi_d_target

    @phi_d_target.setter
    def phi_d_target(self, value):
        self._phi_d_target = value

    @utils.timeIt
    def run(self, m0):
        """run(m0)

            Runs the inversion!

        """
        self.startup(m0)
        while True:
            self.doStartIteration()
            self.m = self.opt.minimize(self.evalFunction, self.m)
            self.doEndIteration()
            if self.stoppingCriteria(): break

        self.printDone()
        self.finish()

        return self.m

    @utils.callHooks('startup')
    def startup(self, m0):
        """
            **startup** is called at the start of any new run call.

            :param numpy.ndarray x0: initial x
            :rtype: None
            :return: None
        """

        if not hasattr(self.reg, '_mref'):
            print 'Regularization has not set mref. SimPEG will set it to m0.'
            self.reg.mref = m0

        self.m = m0
        self._iter = 0
        self._beta = None
        self.phi_d_last = np.nan
        self.phi_m_last = np.nan

    @utils.callHooks('doStartIteration')
    def doStartIteration(self):
        """
            **doStartIteration** is called at the end of each run iteration.

            :rtype: None
            :return: None
        """
        self._beta = self.getBeta()


    @utils.callHooks('doEndIteration')
    def doEndIteration(self):
        """
            **doEndIteration** is called at the end of each run iteration.

            :rtype: None
            :return: None
        """
        # store old values
        self.phi_d_last = self.phi_d
        self.phi_m_last = self.phi_m
        self._iter += 1

    def getBeta(self):
        return self.beta0

    def estimateBeta0(self, u=None, ratio=0.1):
        """estimateBeta0(u=None, ratio=0.1)

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
        if u is None:
            u = self.prob.field(self.m)

        x0 = np.random.rand(*self.m.shape)
        t = x0.dot(self.dataObj2Deriv(self.m,x0,u=u))
        b = x0.dot(self.reg.modelObj2Deriv()*x0)
        return ratio*(t/b)

    def stoppingCriteria(self):
        if self.debug: print 'checking stoppingCriteria'
        return utils.checkStoppers(self, self.stoppers)


    def printDone(self):
        """
            **printDone** is called at the end of the inversion routine.

        """
        utils.printStoppers(self, self.stoppers)

    @utils.callHooks('finish')
    def finish(self):
        """finish()

            **finish** is called at the end of the optimization.
        """
        pass

    @utils.timeIt
    def evalFunction(self, m, return_g=True, return_H=True):
        """evalFunction(m, return_g=True, return_H=True)


        """

        u = self.prob.field(m)

        if self._iter is 0 and self._beta is None:
            self._beta = self.beta0 = self.estimateBeta0(u=u,ratio=self.beta0_ratio)

        phi_d = self.dataObj(m, u)
        phi_m = self.reg.modelObj(m)

        self.dpred = self.prob.dpred(m, u=u)  # This is a cheap matrix vector calculation.
        self.phi_d = phi_d
        self.phi_m = phi_m

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

    @utils.timeIt
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
        R = self.Wd*self.prob.dataResidual(m, u=u)
        R = utils.mkvc(R)
        return 0.5*np.vdot(R, R)

    @utils.timeIt
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
        if u is None:
            u = self.prob.field(m)

        R = self.Wd*self.prob.dataResidual(m, u=u)

        dmisfit = self.prob.Jt(m, self.Wd * R, u=u)

        return dmisfit

    @utils.timeIt
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
        if u is None:
            u = self.prob.field(m)

        R = self.Wd*self.prob.dataResidual(m, u=u)

        # TODO: abstract to different norms a little cleaner.
        #                                        \/ it goes here. in l2 it is the identity.
        dmisfit = self.prob.Jt_approx(m, self.Wd * self.Wd * self.prob.J_approx(m, v, u=u), u=u)

        return dmisfit

    def save(self, group):
        group.attrs['phi_d'] = self.phi_d
        group.attrs['phi_m'] = self.phi_m
        group.setArray('m', self.m)
        group.setArray('dpred', self.dpred)

class Inversion(Cooling, Remember, BaseInversion):

    maxIter = 10
    name = "SimPEG Inversion"

    def __init__(self, prob, reg, opt, **kwargs):
        BaseInversion.__init__(self, prob, reg, opt, **kwargs)

        self.stoppers.append(StoppingCriteria.phi_d_target_Inversion)

        if StoppingCriteria.phi_d_target_Minimize not in self.opt.stoppers:
            self.opt.stoppers.append(StoppingCriteria.phi_d_target_Minimize)

class TimeSteppingInversion(Remember, BaseInversion):
    """
        A slightly different view on regularization parameters,
        let Beta be viewed as 1/dt, and timestep by updating the
        reference model every optimization iteration.
    """
    maxIter = 1
    name = "Time-Stepping SimPEG Inversion"

    def __init__(self, prob, reg, opt, **kwargs):
        BaseInversion.__init__(self, prob, reg, opt, **kwargs)

        self.stoppers.append(StoppingCriteria.phi_d_target_Inversion)

        if StoppingCriteria.phi_d_target_Minimize not in self.opt.stoppers:
            self.opt.stoppers.append(StoppingCriteria.phi_d_target_Minimize)

    def _startup_TimeSteppingInversion(self, m0):

        def _doEndIteration_updateMref(self, xt):
            if self.debug: 'Updating the reference model.'
            self.parent.reg.mref = self.xc

        self.opt.hook(_doEndIteration_updateMref, overwrite=True)
