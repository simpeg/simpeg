from SimPEG import utils


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


class Data(object):
    """Data holds the observed data, and the standard deviations."""

    __metaclass__ = utils.Save.Savable

    std = None    #: Estimated Standard Deviations
    dobs = None   #: Observed data
    dtrue = None  #: True data, if data is synthetic
    mtrue = None  #: True model, if data is synthetic
    prob = None   #: The geophysical problem that explains this data


    def __init__(self, **kwargs):
        utils.setKwargs(self, **kwargs)

    def isSynthetic(self):
        "Check if the data is synthetic."
        return (self.mtrue is not None)

    def setProblem(self, prob):
        self.prob = prob

    @property
    def Wd(self):
        """
            Standard deviation weighting matrix.

            By default, this is based on the norm of the data plus a noise floor.

        """
        if getattr(self,'_Wd',None) is None:
            eps = np.linalg.norm(utils.mkvc(self.dobs),2)*1e-5
            self._Wd = 1/(abs(self.dobs)*self.std+eps)
        return self._Wd
    @Wd.setter
    def Wd(self, value):
        self._Wd = value

    @requiresProblem
    def dpred(self, m, u=None):
        if u is None: u = self.prob.field(m)

    @requiresProblem
    def residual(self, m, u=None):
        if u is None: u = self.prob.field(m)

    @requiresProblem
    def residualWeighted(self, m, u=None):
        if u is None: u = self.prob.field(m)

    @requiresProblem
    def projectField(self, m, u=None):
        """
            Projection matrix.

            .. math::
                d_\\text{pred} = Pu(m)
        """
        if u is None: u = self.prob.field(m)
        return self.P*u


if __name__ == '__main__':
    d = SimPEGData()
    d.dpred()
