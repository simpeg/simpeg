import Utils, Parameters, numpy as np, scipy.sparse as sp

class BaseObjFunction(object):
    """BaseObjFunction(data, reg, **kwargs)"""

    __metaclass__ = Utils.Save.Savable

    beta    = Parameters.ParameterProperty('beta', default=1, doc='Regularization trade-off parameter')

    debug   = False  #: Print debugging information
    counter = None   #: Set this to a SimPEG.Utils.Counter() if you want to count things

    name = 'Base Objective Function'   #: Name of the objective function

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

    @property
    def inv(self): return self.parent
    @property
    def objFunc(self): return self
    @property
    def opt(self): return getattr(self.parent,'opt',None)
    @property
    def prob(self): return self.data.prob
    @property
    def mesh(self): return self.data.prob.mesh
    @property
    def model(self): return self.data.prob.model


    def __init__(self, data, reg, **kwargs):
        Utils.setKwargs(self, **kwargs)

        self.data = data

        self.reg = reg
        self.reg.parent = self


    @Utils.callHooks('startup')
    def startup(self, m0):
        """startup(m0)

            Called when inversion is first starting.
        """
        if self.debug: print 'Calling ObjFunction.startup'

        if self.reg.mref is None:
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

        u = self.data.prob.fields(m)
        self.u_current = u

        phi_d = self.dataObj(m, u=u)
        phi_m = self.reg.modelObj(m)

        self.dpred = self.data.dpred(m, u=u)  # This is a cheap matrix vector calculation.

        self.phi_d, self.phi_d_last  = phi_d, self.phi_d
        self.phi_m, self.phi_m_last  = phi_m, self.phi_m

        f = phi_d + self.beta * phi_m

        out = (f,)
        if return_g:
            phi_dDeriv = self.dataObjDeriv(m, u=u)
            phi_mDeriv = self.reg.modelObjDeriv(m)

            g = phi_dDeriv + self.beta * phi_mDeriv
            out += (g,)

        if return_H:
            def H_fun(v):
                phi_d2Deriv = self.dataObj2Deriv(m, v, u=u)
                phi_m2Deriv = self.reg.modelObj2Deriv()*v

                return phi_d2Deriv + self.beta * phi_m2Deriv

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
        if u is None: u = self.data.prob.fields(m)

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
        if u is None: u = self.data.prob.fields(m)

        R = self.data.residualWeighted(m, u=u)

        # TODO: abstract to different norms a little cleaner.
        #                                                 \/ it goes here. in l2 it is the identity.
        dmisfit = self.data.prob.Jt_approx(m, self.data.Wd * self.data.Wd * self.data.prob.J_approx(m, v, u=u), u=u)

        return dmisfit
