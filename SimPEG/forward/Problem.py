import numpy as np
from SimPEG.utils import mkvc, sdiag
norm = np.linalg.norm


class Problem(object):
    """Problem is the base class for all geophysical forward problems in SimPEG"""
    def __init__(self, mesh):
        self.mesh = mesh
        pass

    def residual(self, m):
        pass

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

    def misfit(self, field):
        """
            :param numpy.array field: geophysical field of interest
            :rtype: float
            :return: data misfit

            The data misfit using an l_2 norm is:

            .. math::

                \mu_\\text{data} = {1\over 2}\left| \mathbf{W} (\mathbf{Pu} - d_\\text{obs}) \\right|_2^2

            Where P is a projection matrix that brings the field on the full domain to the data measurement locations;
            u is the field of interest; d_obs is the observed data; and W is the weighting matrix.
        """
        R = self.W*(self.P*field - self.dobs)
        return 0.5*mkvc(R).inner(mkvc(R))

    def misfitDeriv(self, field):
        """
            TODO: Change this documentation.

            :param numpy.array field: geophysical field of interest
            :rtype: float
            :return: data misfit derivative

            The data misfit using an l_2 norm is:

            .. math::

                \mu_\\text{data} = {1\over 2}\left| \mathbf{W} (\mathbf{Pu} - d_\\text{obs}) \\right|_2^2

            Where P is a projection matrix that brings the field on the full domain to the data measurement locations;
            u is the field of interest; d_obs is the observed data; and W is the weighting matrix.
        """

        R = self.W*(self.P*field - self.dobs)
        # TODO: make in terms of the field and call Jt, e.g. if looping over RHSs using i: self.Jt(field[:,i],self.W[:,i]*R[:,i])
        return mkvc(R)

    def J(self, u):
        pass

    def Jt(self, v):
        pass

if __name__ == '__main__':
    from SimPEG.inverse import checkDerivative

    p = Problem(None)
    m = np.random.rand(5)
    checkDerivative(lambda m : [p.modelTransform(m), p.modelTransformDeriv(m)], m)
