import numpy as np
import scipy.sparse as sp
import SimPEG
from SimPEG.utils import sdiag, mkvc, setKwargs, checkStoppers

class Inversion(object):
    """docstring for Inversion"""

    maxIter = 10
    name = 'SimPEG Inversion'

    def __init__(self, prob, reg, opt, **kwargs):
        setKwargs(self, **kwargs)
        self.prob = prob
        self.reg = reg
        self.opt = opt
        self.opt.parent = self

        self.stoppers = [SimPEG.inverse.StoppingCriteria.iteration, SimPEG.inverse.StoppingCriteria.phi_d_target_Inversion]

        # Check if we have inserted printers into the optimization
        if not np.any([p is SimPEG.inverse.IterationPrinters.phi_d for p in self.opt.printers]):
            self.opt.printers.insert(1,SimPEG.inverse.IterationPrinters.beta)
            self.opt.printers.insert(2,SimPEG.inverse.IterationPrinters.phi_d)
            self.opt.printers.insert(3,SimPEG.inverse.IterationPrinters.phi_m)
            self.opt.stoppers.append(SimPEG.inverse.StoppingCriteria.phi_d_target_Minimize)

    @property
    def Wd(self):
        """
            Standard deviation weighting matrix.
        """
        if getattr(self,'_Wd',None) is None:
            eps = np.linalg.norm(mkvc(self.prob.dobs),2)*1e-5
            self._Wd = 1/(abs(self.prob.dobs)*self.prob.std+eps)
        return self._Wd

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

    def run(self, m0):
        m = m0
        self._iter = 0
        self._beta = None
        while True:
            self._beta = self.getBeta()
            m = self.opt.minimize(self.evalFunction,m)
            self._iter += 1
            if self.stoppingCriteria(): break
        return m

    beta0 = 1.e2
    beta_coolingFactor = 5.

    def getBeta(self):
        if self._beta is None:
            return self.beta0
        return self._beta / self.beta_coolingFactor

    def stoppingCriteria(self):
        return checkStoppers(self, self.stoppers)

    def evalFunction(self, m, return_g=True, return_H=True):

        u = self.prob.field(m)
        phi_d = self.dataObj(m, u)
        phi_m = self.reg.modelObj(m)

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
                phi_m2Deriv = self.reg.modelObj2Deriv(m)*v

                return phi_d2Deriv + self._beta * phi_m2Deriv

            operator = sp.linalg.LinearOperator( (m.size, m.size), H_fun, dtype=float )
            out += (operator,)
        return out if len(out) > 1 else out[0]


    def dataObj(self, m, u=None):
        """
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
        R = mkvc(R)
        return 0.5*np.vdot(R, R)

    def dataObjDeriv(self, m, u=None):
        """
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

    def dataObj2Deriv(self, m, v, u=None):
        """
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

                \\frac{\partial^2 \mu_\\text{data}}{\partial^2 \mathbf{m}} = \mathbf{J}^\\top \mathbf{W \circ W J}

        """
        if u is None:
            u = self.prob.field(m)

        R = self.Wd*self.prob.dataResidual(m, u=u)

        # TODO: abstract to different norms a little cleaner.
        #                                 \/ it goes here. in l2 it is the identity.
        dmisfit = self.prob.Jt_approx(m, self.Wd * self.Wd * self.prob.J_approx(m, v, u=u), u=u)

        return dmisfit

