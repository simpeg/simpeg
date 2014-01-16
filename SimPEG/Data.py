import Utils


def requiresProblem(f):
    """
    Use this to wrap a funciton::

        @requiresProblem
        def dpred(self):
            pass

    This wrapper will ensure that a problem has been bound to the data.
    If a problem is not bound an Exception will be raised, and an nice error message printed.
    """
    extra = """
        This function requires that a problem be bound to the data.
        If a problem has not been bound, an Exception will be raised.
        To bind a problem to the Data object::

            data.setProblem(myProblem)
    """
    from functools import wraps
    @wraps(f)
    def requiresProblemWrapper(self,*args,**kwargs):
        if getattr(self, 'prob', None) is None:
            raise Exception(extra)
        return f(self,*args,**kwargs)

    doc = requiresProblemWrapper.__doc__
    requiresProblemWrapper.__doc__ = ('' if doc is None else doc) + extra

    return requiresProblemWrapper


class BaseData(object):
    """Data holds the observed data, and the standard deviations."""

    __metaclass__ = Utils.Save.Savable

    std = None       #: Estimated Standard Deviations
    dobs = None      #: Observed data
    dtrue = None     #: True data, if data is synthetic
    mtrue = None     #: True model, if data is synthetic
    prob = None      #: The geophysical problem that explains this data

    counter = None   #: A SimPEG.Utils.Counter object

    def __init__(self, **kwargs):
        Utils.setKwargs(self, **kwargs)

    def setProblem(self, prob):
        self.prob = prob

    @Utils.count
    @requiresProblem
    def dpred(self, m, u=None):
        """
            Projection matrix.

            .. math::
                d_\\text{pred} = Pu(m)
        """
        if u is None: u = self.prob.field(m)
        return self.P*u

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
        return self.dpred(m, u=u) - self.dobs


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
        return self.Wd*self.residual(m, u=u)

    @property
    def RHS(self):
        """
            Source matrix.
        """
        return self._RHS
    @RHS.setter
    def RHS(self, value):
        self._RHS = value

    def isSynthetic(self):
        "Check if the data is synthetic."
        return (self.mtrue is not None)

if __name__ == '__main__':
    d = BaseData()
    d.dpred()
