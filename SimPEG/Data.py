import Utils, numpy as np


class BaseData(object):
    """Data holds the observed data, and the standard deviations."""

    __metaclass__ = Utils.SimPEGMetaClass

    std = None       #: Estimated Standard Deviations
    dobs = None      #: Observed data
    dtrue = None     #: True data, if data is synthetic
    mtrue = None     #: True model, if data is synthetic

    counter = None   #: A SimPEG.Utils.Counter object

    def __init__(self, **kwargs):
        Utils.setKwargs(self, **kwargs)

    @property
    def prob(self):
        """
        The geophysical problem that explains this data, use::

            data.pair(prob)
        """
        return getattr(self, '_prob', None)

    def pair(self, p):
        """Bind a problem to this data instance using pointers"""
        assert hasattr(p, 'dataPair'), "Problem must have an attribute 'dataPair'."
        assert isinstance(self, p.dataPair), "Problem requires data object must be an instance of a %s class."%(p.dataPair.__name__)
        if p.ispaired:
            raise Exception("The problem object is already paired to a data. Use prob.unpair()")
        self._prob = p
        p._data = self

    def unpair(self):
        """Unbind a problem from this data instance"""
        if not self.ispaired: return
        self.prob._data = None
        self._prob = None

    @property
    def ispaired(self): return self.prob is not None

    @Utils.count
    @Utils.requires('prob')
    def dpred(self, m, u=None):
        """
            Create the projected data from a model.
            The field, u, (if provided) will be used for the predicted data
            instead of recalculating the fields (which may be expensive!).

            .. math::
                d_\\text{pred} = P(u(m))

            Where P is a projection of the fields onto the data space.
        """
        if u is None: u = self.prob.fields(m)
        return Utils.mkvc(self.projectFields(u))


    @Utils.count
    def projectFields(self, u):
        """
            This function projects the fields onto the data space.


            .. math::
                d_\\text{pred} = \mathbf{P} u(m)
        """
        return u


    @Utils.count
    def projectFieldsAdjoint(self, d):
        """
            This function is the adjoint of the projection.
            **projectFieldsAdjoint** is used in the
            calculation of the sensitivities.

            .. math::
                u = \mathbf{P}^\\top d

            :param numpy.array d: data
            :param numpy.array u: fields (ish)
            :rtype: fields like object
            :return: data
        """
        return d

    #TODO: def projectFieldDeriv(self, u):  Does this need to be made??!

    @Utils.count
    def residual(self, m, u=None):
        """
            :param numpy.array m: geophysical model
            :param numpy.array u: fields
            :rtype: float
            :return: data residual

            The data residual:

            .. math::

                \mu_\\text{data} = \mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs}

        """
        return Utils.mkvc(self.dpred(m, u=u) - self.dobs)


    @property
    def Wd(self):
        """
            Data weighting matrix. This is a covariance matrix used in::

                def data.residualWeighted(m,u=None):
                    return self.Wd*self.residual(m, u=u)

            By default, this is based on the norm of the data plus a noise floor.

        """
        if getattr(self,'_Wd',None) is None:
            eps = np.linalg.norm(Utils.mkvc(self.dobs),2)*1e-5
            self._Wd = 1/(abs(self.dobs)*self.std+eps)
        return self._Wd
    @Wd.setter
    def Wd(self, value):
        self._Wd = value

    def residualWeighted(self, m, u=None):
        """
            :param numpy.array m: geophysical model
            :param numpy.array u: fields
            :rtype: float
            :return: data residual

            The weighted data residual:

            .. math::

                \mu_\\text{data}^{\\text{weighted}} = \mathbf{W}_d(\mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs})

            Where W_d is a covariance matrix that weights the data residual.
        """
        return Utils.mkvc(self.Wd*self.residual(m, u=u))

    @property
    def RHS(self):
        """
            Source matrix.
        """
        return getattr(self, '_RHS', None)
    @RHS.setter
    def RHS(self, value):
        self._RHS = value

    @property
    def isSynthetic(self):
        "Check if the data is synthetic."
        return (self.mtrue is not None)

if __name__ == '__main__':
    d = BaseData()
    d.dpred()
