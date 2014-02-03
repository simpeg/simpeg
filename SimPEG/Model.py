import Utils, Parameters, numpy as np, scipy.sparse as sp


class BaseModel(object):
    """
    SimPEG Model

    """

    __metaclass__ = Utils.Save.Savable

    counter = None   #: A SimPEG.Utils.Counter object
    mesh = None      #: A SimPEG Mesh

    def __init__(self, mesh):
        self.mesh = mesh

    def transform(self, m):
        """
            :param numpy.array m: model
            :rtype: numpy.array
            :return: transformed model

            The *transform* changes the model into the physical property.

        """
        return m

    def transformInverse(self, D):
        """
            :param numpy.array D: physical property
            :rtype: numpy.array
            :return: model

            The *transformInverse* changes the physical property into the model.

            .. note:: The *transformInverse* may not be easy to create in general.

        """
        raise NotImplementedError('The transformInverse is not implemented.')

    def transformDeriv(self, m):
        """
            :param numpy.array m: model
            :rtype: scipy.csr_matrix
            :return: derivative of transformed model

            The *transform* changes the model into the physical property.
            The *transformDeriv* provides the derivative of the *transform*.
        """
        return sp.identity(m.size)

    @property
    def nP(self):
        """Number of parameters in the model."""
        return self.mesh.nC

    def example(self, modelType=None):
        return np.random.rand(self.mesh.nC)



class LogModel(BaseModel):
    """SimPEG LogModel"""

    def __init__(self, mesh, **kwargs):
        BaseModel.__init__(self, mesh, **kwargs)

    def transform(self, m):
        """
            :param numpy.array m: model
            :rtype: numpy.array
            :return: transformed model

            The *transform* changes the model into the physical property.

            A common example of this is to invert for electrical conductivity
            in log space. In this case, your model will be log(sigma) and to
            get back to sigma, you can take the exponential:

            .. math::

                m = \log{\sigma}

                \exp{m} = \exp{\log{\sigma}} = \sigma
        """
        return np.exp(Utils.mkvc(m))


    def transformInverse(self, D):
        """
            :param numpy.array D: physical property
            :rtype: numpy.array
            :return: model

            The *transformInverse* changes the physical property into the model.

            .. math::

                m = \log{\sigma}

        """
        return np.log(Utils.mkvc(D))


    def transformDeriv(self, m):
        """
            :param numpy.array m: model
            :rtype: scipy.csr_matrix
            :return: derivative of transformed model

            The *transform* changes the model into the physical property.
            The *transformDeriv* provides the derivative of the *transform*.

            If the model *transform* is:

            .. math::

                m = \log{\sigma}

                \exp{m} = \exp{\log{\sigma}} = \sigma

            Then the derivative is:

            .. math::

                \\frac{\partial \exp{m}}{\partial m} = \\text{sdiag}(\exp{m})
        """
        return Utils.sdiag(np.exp(Utils.mkvc(m)))
