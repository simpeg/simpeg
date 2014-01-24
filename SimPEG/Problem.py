import Utils, Data, numpy as np, scipy.sparse as sp


class BaseProblem(object):
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

    __metaclass__ = Utils.Save.Savable

    counter = None   #: A SimPEG.Utils.Counter object

    dataPair = Data.BaseData

    def __init__(self, mesh, model, *args, **kwargs):
        Utils.setKwargs(self, **kwargs)
        self.mesh = mesh
        self.model = model

    @property
    def data(self):
        """
        The data object for this problem.
        """
        return getattr(self, '_data', None)

    def pair(self, d):
        """Bind a data to this problem instance using pointers."""
        assert isinstance(d, self.dataPair), "Data object must be an instance of a %s class."%(self.dataPair.__name__)
        if d.ispaired:
            raise Exception("The data object is already paired to a problem. Use data.unpair()")
        self._data = d
        d._prob = self

    def unpair(self):
        """Unbind a data from this problem instance."""
        if not self.ispaired: return
        self.data._prob = None
        self._data = None

    @property
    def ispaired(self): return self.data is not None

    @Utils.timeIt
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

    @Utils.timeIt
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


    @Utils.timeIt
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

    @Utils.timeIt
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

    def createSyntheticData(self, m, std=0.05, u=None, **geometry_kwargs):
        """
            Create synthetic data given a model, and a standard deviation.

            :param numpy.array m: geophysical model
            :param numpy.array std: standard deviation
            :rtype: numpy.array, numpy.array
            :return: dobs, Wd

            Returns the observed data with random Gaussian noise
            and Wd which is the same size as data, and can be used to weight the inversion.
        """
        data = self.dataPair(mtrue=m, **geometry_kwargs)
        data.pair(self)
        data.dtrue = data.dpred(m, u=u)
        noise = std*abs(data.dtrue)*np.random.randn(*data.dtrue.shape)
        data.dobs = data.dtrue+noise
        data.std = data.dobs*0 + std
        return data



