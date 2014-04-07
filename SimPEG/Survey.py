import Utils, numpy as np


class BaseSurvey(object):
    """Survey holds the observed data, and the standard deviations."""

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
        The geophysical problem that explains this survey, use::

            survey.pair(prob)
        """
        return getattr(self, '_prob', None)

    @property
    def mesh(self):
        """Mesh of the paired problem."""
        if self.ispaired:
            return self.prob.mesh
        raise Exception('Pair survey to a problem to access the problems mesh.')

    def pair(self, p):
        """Bind a problem to this survey instance using pointers"""
        assert hasattr(p, 'surveyPair'), "Problem must have an attribute 'surveyPair'."
        assert isinstance(self, p.surveyPair), "Problem requires survey object must be an instance of a %s class."%(p.surveyPair.__name__)
        if p.ispaired:
            raise Exception("The problem object is already paired to a survey. Use prob.unpair()")
        self._prob = p
        p._survey = self

    def unpair(self):
        """Unbind a problem from this survey instance"""
        if not self.ispaired: return
        self.prob._survey = None
        self._prob = None

    @property
    def nD(self):
        """Number of data."""
        if hasattr(self, 'dobs'):
            return self.dobs.size
        raise NotImplemented('Number of data is unknown.')

    @property
    def ispaired(self): return self.prob is not None

    @Utils.count
    @Utils.requires('prob')
    def dpred(self, m, u=None):
        """dpred(m, u=None)

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
        """projectFields(u)

            This function projects the fields onto the data space.

            .. math::

                d_\\text{pred} = \mathbf{P} u(m)
        """
        raise NotImplemented('projectFields is not yet implemented.')

    @Utils.count
    def projectFieldsDeriv(self, u):
        """projectFieldsDeriv(u)

            This function s the derivative of projects the fields onto the data space.

            .. math::

                \\frac{\partial d_\\text{pred}}{\partial u} = \mathbf{P}
        """
        raise NotImplemented('projectFields is not yet implemented.')

    @Utils.count
    def residual(self, m, u=None):
        """residual(m, u=None)

            :param numpy.array m: geophysical model
            :param numpy.array u: fields
            :rtype: numpy.array
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

                def residualWeighted(m,u=None):
                    return self.Wd*self.residual(m, u=u)

            By default, this is based on the norm of the data plus a noise floor.

        """
        if getattr(self,'_Wd',None) is None:
            print 'SimPEG is making Survey.Wd to be norm of the data plus a floor.'
            eps = np.linalg.norm(Utils.mkvc(self.dobs),2)*1e-5
            self._Wd = 1/(abs(self.dobs)*self.std+eps)
        return self._Wd
    @Wd.setter
    def Wd(self, value):
        self._Wd = value

    def residualWeighted(self, m, u=None):
        """residualWeighted(m, u=None)

            :param numpy.array m: geophysical model
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: weighted data residual

            The weighted data residual:

            .. math::

                \mu_\\text{data}^{\\text{weighted}} = \mathbf{W}_d(\mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs})

            Where \\\\(W_d\\\\) is a covariance matrix that weights the data residual.
        """
        return Utils.mkvc(self.Wd*self.residual(m, u=u))

    @property
    def isSynthetic(self):
        "Check if the data is synthetic."
        return self.mtrue is not None


    #TODO: Move this to the survey class?
    # @property
    # def phi_d_target(self):
    #     """
    #     target for phi_d

    #     By default this is the number of data.

    #     Note that we do not set the target if it is None, but we return the default value.
    #     """
    #     if getattr(self, '_phi_d_target', None) is None:
    #         return self.data.dobs.size #
    #     return self._phi_d_target

    # @phi_d_target.setter
    # def phi_d_target(self, value):
    #     self._phi_d_target = value


class BaseRx(object):
    """SimPEG Receiver Object"""

    locs = None   #: Locations (nRx x 3)

    knownRxTypes = None  #: Set this to a list of strings to ensure that txType is known

    def __init__(self, locs, rxType, **kwargs):
        self.locs = locs
        self.rxType = rxType
        Utils.setKwargs(self, **kwargs)

    @property
    def rxType(self):
        """Receiver Type"""
        return getattr(self, '_rxType', None)
    @rxType.setter
    def rxType(self, value):
        known = self.knownRxTypes
        if known is not None:
            assert value in known, "rxType must be in ['%s']" % ("', '".join(known))
        self._rxType = value

    @property
    def nD(self):
        return self.locs.shape[0]

class BaseTx(object):
    """SimPEG Transmitter Object"""

    loc    = None #: Location [x,y,z]

    rxList = None #: SimPEG Receiver List
    rxPair = BaseRx

    knownTxTypes = None #: Set this to a list of strings to ensure that txType is known

    def __init__(self, loc, txType, rxList, **kwargs):
        assert type(rxList) is list, 'rxList must be a list'
        for rx in rxList:
            assert isinstance(rx, self.rxPair), 'rxList must be a %s'%self.rxListPair.__name__
        assert len(set(rxList)) == len(rxList), 'The rxList must be unique'

        self.loc    = loc
        self.txType = txType
        self.rxList = rxList
        Utils.setKwargs(self, **kwargs)

    @property
    def txType(self):
        """Transmitter Type"""
        return getattr(self, '_txType', None)
    @txType.setter
    def txType(self, value):
        known = self.knownTxTypes
        if known is not None:
            assert value in known, "txType must be in ['%s']" % ("', '".join(known))
        self._txType = value

if __name__ == '__main__':
    d = BaseData()
    d.dpred()
