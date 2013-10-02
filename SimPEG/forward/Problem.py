import numpy as np
from SimPEG.utils import mkvc, sdiag
norm = np.linalg.norm


class Problem(object):
    """
        Problem is the base class for all geophysical forward problems in SimPEG.


        The problem is a partial differential equation of the form:

        .. math::
            c(m, u) = 0

        Here, m is the model and u is the field (or fields).
        Given the model, m, we can calculate the fields u(m),
        however, the data we collect is a subset of the fields,
        and can be defined by a linear projection, P.

        .. math::
            d_\\text{pred} = Pu(m)

        We are interested in how changing the model transforms the data,
        as such we can take write the Taylor expansion:

        .. math::
            Pu(m + hv) = Pu(m) + hP\\frac{\partial u(m)}{\partial m} v + \mathcal{O}(h^2 \left\| v \\right\| )

        We can linearize and define the sensitivity matrix as:

        .. math::
            J = P\\frac{\partial u}{\partial m}

        The sensitivity matrix, and it's transpose will be used in the inverse problem
        to (locally) find how model parameters change the data, and optimize!
    """

    def __init__(self, mesh):
        self.mesh = mesh

    @property
    def RHS(self):
        """
            Source matrix.
        """
        return self._RHS
    @RHS.setter
    def RHS(self, value):
        self._RHS = value

    @property
    def W(self):
        """
            Standard deviation weighting matrix.
        """
        return self._W
    @W.setter
    def W(self, value):
        self._W = value

    @property
    def P(self):
        """
            Projection matrix.

            .. math::
                d_\\text{pred} = Pu(m)
        """
        return self._P
    @P.setter
    def P(self, value):
        self._P = value


    @property
    def dobs(self):
        """
            Observed data.
        """
        return self._dobs
    @dobs.setter
    def dobs(self, value):
        self._P = value


    def J(self, u):
        """
            Working with the general PDE, c(m, u) = 0, where m is the model and u is the field,
            the sensitivity is defined as:

            .. math::
                J = P\\frac{\partial u}{\partial m}

            We can take the derivative of the PDE:

            .. math::
                \\nabla_m c(m, u) \delta m + \\nabla_u c(m, u) \delta u = 0

            If the forward problem is invertible, then we can rearrange for du/dm:

            .. math::
                J = - P \left( \\nabla_u c(m, u) \\right)^{-1} \\nabla_m c(m, u)

            This can often be computed given a vector (i.e. J(v)) rather than stored, as J is a large dense matrix.

        """
        pass

    def Jt(self, v):
        """
            Transpose of J
        """
        pass

    def field(self, m):
        """
            The fields.
        """
        pass

    def dpred(self, m, u=None):
        """
            Predicted data.

            .. math::
                d_\\text{pred} = Pu(m)
        """
        if u is None:
            u = self.field(m)
        return self.P*u

    def modelTransform(self, m):
        """
            :param numpy.array m: model
            :rtype: numpy.array
            :return: transformed model

            The modelTransform changes the model into the physical property.

            A common example of this is to invert for electrical conductivity
            in log space. In this case, your model will be log(sigma) and to
            get back to sigma, you can take the exponential:

            .. math::

                m = \log{\sigma}

                \exp{m} = \exp{\log{\sigma}} = \sigma
        """
        return np.exp(mkvc(m))

    def modelTransformDeriv(self, m):
        """
            :param numpy.array m: model
            :rtype: scipy.csr_matrix
            :return: derivative of transformed model

            The modelTransform changes the model into the physical property.
            The modelTransformDeriv provides the derivative of the modelTransform.

            If the model transform is:

            .. math::

                m = \log{\sigma}

                \exp{m} = \exp{\log{\sigma}} = \sigma

            Then the derivative is:

            .. math::

                \\frac{\partial \exp{m}}{\partial m} = \\text{sdiag}(\exp{m})
        """
        return sdiag(np.exp(mkvc(m)))

    def _test_modelTransformDeriv(self):
        m = np.random.rand(5)
        return checkDerivative(lambda m : [self.modelTransform(m), self.modelTransformDeriv(m)], m)

    def misfit(self, m, R=None):
        """
            :param numpy.array m: geophysical model
            :param numpy.array R: residual, R = W o (dpred - dobs)
            :rtype: float
            :return: data misfit

            The data misfit using an l_2 norm is:

            .. math::

                \mu_\\text{data} = {1\over 2}\left| \mathbf{W} \circ (\mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs}) \\right|_2^2

            Where P is a projection matrix that brings the field on the full domain to the data measurement locations;
            u is the field of interest; d_obs is the observed data; and W is the weighting matrix.
        """
        if R is None:
            R = self.W*(self.dpred(m) - self.dobs)

        R = mkvc(R)
        return 0.5*R.inner(R)

    def misfitDeriv(self, m, R=None, u=None):
        """
            :param numpy.array m: geophysical model
            :rtype: numpy.array
            :return: data misfit derivative

            The data misfit using an l_2 norm is:

            .. math::

                \mu_\\text{data} = {1\over 2}\left| \mathbf{W} \circ (\mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs}) \\right|_2^2

                \mathbf{R} = \mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs}

                \mu_\\text{data} = {1\over 2}\left| \mathbf{W \circ R} \\right|_2^2

            Where P is a projection matrix that brings the field on the full domain to the data measurement locations;
            u is the field of interest; d_obs is the observed data; and W is the weighting matrix.

            The derivative of this, with respect to the model, is:

            .. math::

                \\frac{\partial \mu_\\text{data}}{\partial \mathbf{m}} = \mathbf{J}^\\top (\mathbf{W \circ R})

        """
        if u is None:
            u = self.field(m)

        if R is None:
            R = self.W*(self.dpred(m, u=u) - self.dobs)

        dmisfit = 0
        for i in range(self.RHS.shape[1]): # Loop over each right hand side
            dmisfit += self.Jt(u[:,i], self.W[:,i]*R[:,i])

        return dmisfit


if __name__ == '__main__':
    from SimPEG.inverse import checkDerivative

    p = Problem(None)
    m = np.random.rand(5)
    checkDerivative(lambda m : [p.modelTransform(m), p.modelTransformDeriv(m)], m)
