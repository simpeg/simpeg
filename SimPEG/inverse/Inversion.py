import numpy as np

class Inversion(object):
    """docstring for Inversion"""

    maxIter = 10

    def __init__(self, prob, reg, opt):
        self.prob = prob
        self.reg = reg
        self.opt = opt

    @property
    def W(self):
        """
            Standard deviation weighting matrix.
        """
        return self._W
    @W.setter
    def W(self, value):
        self._W = value

    def run(self, m0):
        self._iter = 0
        while True:
            self._beta = self.getBeta()
            self.opt.minimize(self.evalFunction,m)
            if self.stoppingCriteria(): break
            self._iter += 1

    def getBeta(self):
        return 1

    def stoppingCriteria(self):
        self._STOP = np.zeros(2,dtype=bool)
        self._STOP[0] = self._iter >= maxIter
        self._STOP[1] = self._phi_d_last <= self.phi_d_target
        return np.any(self._STOP)


    def evalFunction(self, m, return_g=True, return_H=True):

        u = self.prob.field(m)
        phi_d = self.dataObj(m, u)
        phi_m = self.modelObj(m)

        self._phi_d_last = phi_d
        self._phi_m_last = phi_m

        f = phi_d + self._beta * phi_m

        out = (f,)
        if return_g:
            phi_dDeriv = self.dataObjDeriv(m, u)
            phi_mDeriv = self.modelObjDeriv(m)

            g = phi_dDeriv + self._beta * phi_mDeriv
            out += (g,)

        if return_H:
            def H_fun(v):
                phi_d2Deriv = self.dataObj2Deriv(m, u, v)
                phi_m2Deriv = self.modelObj2Deriv(m)*v

                return phi_d2Deriv + self._beta * phi_m2Deriv

            out += (H_fun,)
        return out


    def modelObj(self, m, u=None):
        self.reg.misfit(m)


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
        R = self.Wd*self.prob.misfit(u=u)
        R = mkvc(R)
        return 0.5*R.dot(R)

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
            u = self.field(m)

        R = self.W*(self.dpred(m, u=u) - self.dobs)

        dmisfit = 0
        for i in range(self.RHS.shape[1]): # Loop over each right hand side
            dmisfit += self.Jt(m, self.W[:,i]*R[:,i], u=u[:,i])

        return dmisfit

    def dataObj2Deriv(self, m, u=None):
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
            u = self.field(m)

        R = self.W*(self.dpred(m, u=u) - self.dobs)

        dmisfit = 0
        for i in range(self.RHS.shape[1]): # Loop over each right hand side
            dmisfit += self.Jt(m, self.W[:,i]*R[:,i], u=u[:,i])

        return dmisfit
