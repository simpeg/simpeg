import Utils, numpy as np, scipy.sparse as sp


class BaseRx(object):
    """SimPEG Receiver Object"""

    locs = None   #: Locations (nRx x nDim)

    knownRxTypes = None  #: Set this to a list of strings to ensure that txType is known

    projGLoc = 'CC'  #: Projection grid location, default is CC

    storeProjections = True #: Store calls to getP (organized by mesh)

    def __init__(self, locs, rxType, **kwargs):
        self.locs = locs
        self.rxType = rxType
        self._Ps = {}
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
        """Number of data in the receiver."""
        return self.locs.shape[0]

    def getP(self, mesh):
        """
            Returns the projection matrices as a
            list for all components collected by
            the receivers.

            .. note::

                Projection matrices are stored as a dictionary listed by meshes.
        """
        if mesh in self._Ps:
            return self._Ps[mesh]

        P = mesh.getInterpolationMat(self.locs, self.projGLoc)
        if self.storeProjections:
            self._Ps[mesh] = P
        return P


class BaseTimeRx(BaseRx):
    """SimPEG Receiver Object"""

    times = None   #: Times when the receivers were active.
    projTLoc = 'N'

    def __init__(self, locs, times, rxType, **kwargs):
        self.times = times
        BaseRx.__init__(self, locs, rxType, **kwargs)

    @property
    def nD(self):
        """Number of data in the receiver."""
        return self.locs.shape[0] * len(self.times)

    def getSpatialP(self, mesh):
        """
            Returns the spatial projection matrix.

            .. note::

                This is not stored in memory, but is created on demand.
        """
        return mesh.getInterpolationMat(self.locs, self.projGLoc)

    def getTimeP(self, timeMesh):
        """
            Returns the time projection matrix.

            .. note::

                This is not stored in memory, but is created on demand.
        """
        return timeMesh.getInterpolationMat(self.times, self.projTLoc)

    def getP(self, mesh, timeMesh):
        """
            Returns the projection matrices as a
            list for all components collected by
            the receivers.

            .. note::

                Projection matrices are stored as a dictionary (mesh, timeMesh) if storeProjections is True
        """
        if (mesh, timeMesh) in self._Ps:
            return self._Ps[(mesh, timeMesh)]

        Ps = self.getSpatialP(mesh)
        Pt = self.getTimeP(timeMesh)
        P = sp.kron(Pt, Ps)

        if self.storeProjections:
            self._Ps[(mesh, timeMesh)] = P

        return P


class BaseTx(object):
    """SimPEG Transmitter Object"""

    loc    = None #: Location [x,y,z]

    rxList = None #: SimPEG Receiver List
    rxPair = BaseRx

    knownTxTypes = None #: Set this to a list of strings to ensure that txType is known

    def __init__(self, loc, txType, rxList, **kwargs):
        assert type(rxList) is list, 'rxList must be a list'
        for rx in rxList:
            assert isinstance(rx, self.rxPair), 'rxList must be a %s'%self.rxPair.__name__
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

    @property
    def nD(self):
        """Number of data"""
        return self.vnD.sum()

    @property
    def vnD(self):
        """Vector number of data"""
        return np.array([rx.nD for rx in self.rxList])


class Data(object):
    """Fancy data storage by Tx and Rx"""

    def __init__(self, survey, v=None):
        self.survey = survey
        self._dataDict = {}
        for tx in self.survey.txList:
            self._dataDict[tx] = {}
        if v is not None:
            self.fromvec(v)

    def _ensureCorrectKey(self, key):
        if type(key) is tuple:
            if len(key) is not 2:
                raise KeyError('Key must be [Tx, Rx]')
            if key[0] not in self.survey.txList:
                raise KeyError('Tx Key must be a transmitter in the survey.')
            if key[1] not in key[0].rxList:
                raise KeyError('Rx Key must be a receiver for the transmitter.')
            return key
        elif isinstance(key, self.survey.txPair):
            if key not in self.survey.txList:
                raise KeyError('Key must be a transmitter in the survey.')
            return key, None
        else:
            raise KeyError('Key must be [Tx] or [Tx,Rx]')

    def __setitem__(self, key, value):
        tx, rx = self._ensureCorrectKey(key)
        assert rx is not None, 'set data using [Tx, Rx]'
        assert isinstance(value, np.ndarray), 'value must by ndarray'
        assert value.size == rx.nD, "value must have the same number of data as the transmitter."
        self._dataDict[tx][rx] = Utils.mkvc(value)

    def __getitem__(self, key):
        tx, rx = self._ensureCorrectKey(key)
        if rx is not None:
            if rx not in self._dataDict[tx]:
                raise Exception('Data for receiver has not yet been set.')
            return self._dataDict[tx][rx]

        return np.concatenate([self[tx,rx] for rx in tx.rxList])

    def tovec(self):
        return np.concatenate([self[tx] for tx in self.survey.txList])

    def fromvec(self, v):
        v = Utils.mkvc(v)
        assert v.size == self.survey.nD, 'v must have the correct number of data.'
        indBot, indTop = 0, 0
        for tx in self.survey.txList:
            for rx in tx.rxList:
                indTop += rx.nD
                self[tx, rx] = v[indBot:indTop]
                indBot += rx.nD


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

    txPair = BaseTx  #: Transmitter Pair

    @property
    def txList(self):
        """Transmitter List"""
        return getattr(self, '_txList', None)

    @txList.setter
    def txList(self, value):
        assert type(value) is list, 'txList must be a list'
        assert np.all([isinstance(tx, self.txPair) for tx in value]), 'All transmitters must be instances of %s' % self.txPair.__name__
        assert len(set(value)) == len(value), 'The txList must be unique'
        self._txList = value

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
    def ispaired(self): return self.prob is not None

    @property
    def nD(self):
        """Number of data"""
        return self.vnD.sum()

    @property
    def vnD(self):
        """Vector number of data"""
        return np.array([tx.nD for tx in self.txList])

    @property
    def nTx(self):
        """Number of Transmitters"""
        return len(self.txList)

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
    def isSynthetic(self):
        "Check if the data is synthetic."
        return self.mtrue is not None

    def makeSyntheticData(self, m, std=0.05, u=None):
        """
            Make synthetic data given a model, and a standard deviation.

            :param numpy.array m: geophysical model
            :param numpy.array std: standard deviation
            :param numpy.array u: fields for the given model (if pre-calculated)

        """
        if getattr(self, 'dobs', None) is not None:
            raise Exception('Survey already has dobs.')
        self.mtrue = m
        self.dtrue = self.dpred(m, u=u)
        noise = std*abs(self.dtrue)*np.random.randn(*self.dtrue.shape)
        self.dobs = self.dtrue+noise
        self.std = self.dobs*0 + std
