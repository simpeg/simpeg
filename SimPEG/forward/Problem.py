from SimPEG import utils, data, np, sp
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

    __metaclass__ = utils.Save.Savable

    counter = None   #: A SimPEG.utils.Counter object


    def __init__(self, mesh, *args, **kwargs):
        utils.setKwargs(self, **kwargs)
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

    @utils.count
    def dpred(self, m, u=None):
        """
            Predicted data.

            .. math::
                d_\\text{pred} = Pu(m)
        """
        if u is None:
            u = self.field(m)
        return self.P*u

    @utils.count
    def dataResidual(self, m, data, u=None):
        """
            :param numpy.array m: geophysical model
            :param numpy.array u: fields
            :rtype: float
            :return: data misfit

            The data misfit:

            .. math::

                \mu_\\text{data} = \mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs}

            Where P is a projection matrix that brings the field on the full domain to the data measurement locations;
            u is the field of interest; d_obs is the observed data.
        """

        return self.dpred(m, u=u) - data.dobs

    @utils.timeIt
    def J(self, m, v, u=None):
        """
            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: Jv

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
        raise NotImplementedError('J is not yet implemented.')

    @utils.timeIt
    def Jt(self, m, v, u=None):
        """
            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: JTv

            Effect of transpose of J on a vector v.
        """
        raise NotImplementedError('Jt is not yet implemented.')


    @utils.timeIt
    def J_approx(self, m, v, u=None):
        """

            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: Jv

            Approximate effect of J on a vector v

        """
        return self.J(m, v, u)

    @utils.timeIt
    def Jt_approx(self, m, v, u=None):
        """
            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: JTv

            Approximate transpose of J*v

        """
        return self.Jt(m, v, u)

    def field(self, m):
        """
            The field given the model.

            .. math::
                u(m)

        """
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

        """
        return m

    def modelTransformDeriv(self, m):
        """
            :param numpy.array m: model
            :rtype: scipy.csr_matrix
            :return: derivative of transformed model

            The modelTransform changes the model into the physical property.
            The modelTransformDeriv provides the derivative of the modelTransform.
        """
        return sp.identity(m.size)

    def createSyntheticData(self, m, std=0.05, u=None):
        """
            Create synthetic data given a model, and a standard deviation.

            :param numpy.array m: geophysical model
            :param numpy.array std: standard deviation
            :rtype: numpy.array, numpy.array
            :return: dobs, Wd

            Returns the observed data with random Gaussian noise
            and Wd which is the same size as data, and can be used to weight the inversion.
        """
        dtrue = self.dpred(m,u=u)
        noise = std*abs(dtrue)*np.random.randn(*dtrue.shape)
        dobs = dtrue+noise
        stdev = dobs*0 + std
        return data.SimPEGData(self, dobs=dobs, std=stdev, dtrue=dtrue, mtrue=m)



