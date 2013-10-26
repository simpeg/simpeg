import numpy as np
from SimPEG.utils import mkvc, sdiag

class LogModel(object):
    """docstring for LogModel"""
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
