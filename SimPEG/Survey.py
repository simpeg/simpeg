from __future__ import print_function
from . import Utils
from . import Props
import numpy as np
import scipy.sparse as sp
import uuid
import gc


class BaseRx(object):
    """SimPEG Receiver Object"""

    locs = None   #: Locations (nRx x nDim)

    knownRxTypes = None  #: Set this to a list of strings to ensure that srcType is known

    projGLoc = 'CC'  #: Projection grid location, default is CC

    storeProjections = True #: Store calls to getP (organized by mesh)

    def __init__(self, locs, rxType, **kwargs):
        self.uid = str(uuid.uuid4())
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
            assert value in known, (
                "rxType must be in ['{0!s}']".format(("', '".join(known)))
            )
        self._rxType = value

    @property
    def nD(self):
        """Number of data in the receiver."""
        return self.locs.shape[0]

    def getP(self, mesh, projGLoc=None):
        """
            Returns the projection matrices as a
            list for all components collected by
            the receivers.

            .. note::

                Projection matrices are stored as a dictionary listed by meshes.
        """
        if mesh in self._Ps:
            return self._Ps[mesh]

        if projGLoc is None:
            projGLoc = self.projGLoc

        P = mesh.getInterpolationMat(self.locs, projGLoc)
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

                Projection matrices are stored as a dictionary (mesh, timeMesh)
                if storeProjections is True
        """
        if (mesh, timeMesh) in self._Ps:
            return self._Ps[(mesh, timeMesh)]

        Ps = self.getSpatialP(mesh)
        Pt = self.getTimeP(timeMesh)
        P = sp.kron(Pt, Ps)

        if self.storeProjections:
            self._Ps[(mesh, timeMesh)] = P

        return P


class BaseSrc(Props.BaseSimPEG):
    """SimPEG Source Object"""

    loc    = None #: Location [x,y,z]

    rxList = None #: SimPEG Receiver List
    rxPair = BaseRx

    def __init__(self, rxList, **kwargs):
        assert type(rxList) is list, 'rxList must be a list'
        for rx in rxList:
            assert isinstance(rx, self.rxPair), (
                'rxList must be a {0!s}'.format(self.rxPair.__name__)
            )
        assert len(set(rxList)) == len(rxList), 'The rxList must be unique'
        self.uid = str(uuid.uuid4())
        self.rxList = rxList
        Utils.setKwargs(self, **kwargs)


    @property
    def nD(self):
        """Number of data"""
        return self.vnD.sum()

    @property
    def vnD(self):
        """Vector number of data"""
        return np.array([rx.nD for rx in self.rxList])


class BaseData(object):
    """Fancy data storage by Survey's Src and Rx"""

    def __init__(self, survey, v=None):
        self.uid = str(uuid.uuid4())
        self.survey = survey
        self._dataDict = {}
        for src in self.survey.srcList:
            self._dataDict[src] = {}
        if v is not None:
            self.fromvec(v)

    def _ensureCorrectKey(self, key):
        if type(key) is tuple:
            if len(key) is not 2:
                raise KeyError('Key must be [Src, Rx]')
            if key[0] not in self.survey.srcList:
                raise KeyError('Src Key must be a source in the survey.')
            if key[1] not in key[0].rxList:
                raise KeyError('Rx Key must be a receiver for the source.')
            return key
        elif isinstance(key, self.survey.srcPair):
            if key not in self.survey.srcList:
                raise KeyError('Key must be a source in the survey.')
            return key, None
        else:
            raise KeyError('Key must be [Src] or [Src,Rx]')

    def __setitem__(self, key, value):
        src, rx = self._ensureCorrectKey(key)
        assert rx is not None, 'set data using [Src, Rx]'
        assert isinstance(value, np.ndarray), 'value must by ndarray'
        assert value.size == rx.nD, (
            "value must have the same number of data as the source."
        )
        self._dataDict[src][rx] = Utils.mkvc(value)

    def __getitem__(self, key):
        src, rx = self._ensureCorrectKey(key)
        if rx is not None:
            if rx not in self._dataDict[src]:
                raise Exception('Data for receiver has not yet been set.')
            return self._dataDict[src][rx]

        return np.concatenate([self[src,rx] for rx in src.rxList])

    def tovec(self):
        return np.concatenate([self[src] for src in self.survey.srcList])

    def fromvec(self, v):
        v = Utils.mkvc(v)
        assert v.size == self.survey.nD, (
            'v must have the correct number of data.'
        )
        indBot, indTop = 0, 0
        for src in self.survey.srcList:
            for rx in src.rxList:
                indTop += rx.nD
                self[src, rx] = v[indBot:indTop]
                indBot += rx.nD


class Data(BaseData):
    """
    Storage of data, standard_deviation and floor storage
    with fancy [Src,Rx] indexing.

    **Requried**
    :param Survey survey: The survey descriping the layout of the data

    **Optional**
    :param ndarray dobs: The data vector matching the src and rx in survey
    :param ndarray standard_deviation: The standard deviation vector matching the src and rx in survey
    :param ndarray floor: The floor vector for the data matching the src and rx in survey

    """

    def __init__(self, survey, dobs=None, standard_deviation=None, floor=None):
        # Initiate the base problem
        BaseData.__init__(self, survey, dobs)

        # Set the uncertainty parameters
        # Note: Maybe set these
        self.standard_deviation = StandardDeviation(
            self.survey, standard_deviation)
        self.floor = Floor(self.survey, floor)

    def calculate_uncertainty(self):
        """
        Return the uncertainty base on
        standard_devation * np.abs(data) + floor

        """
        return (
            self.standard_deviation.tovec() * np.abs(self.tovec()) +
            self.floor.tovec())


class StandardDeviation(BaseData):
    """
    Storage of standard deviation estimates of data
    With fancy [Src,Rx] indexing.

    """

    def __init__(self, survey, standard_deviation=None):
        # Initiate the base problem
        BaseData.__init__(self, survey, standard_deviation)


class Floor(BaseData):
    """
    Storage of floor estimates of data
    With fancy [Src,Rx] indexing.

    """

    def __init__(self, survey, floor=None):
        # Initiate the base problem
        BaseData.__init__(self, survey, floor)


class BaseSurvey(object):
    """Survey holds the observed data, and the standard deviations."""

    std = None       #: Estimated Standard Deviations
    eps = None       #: Estimated Noise Floor
    dobs = None      #: Observed data
    dtrue = None     #: True data, if data is synthetic
    mtrue = None     #: True model, if data is synthetic

    counter = None   #: A SimPEG.Utils.Counter object

    def __init__(self, **kwargs):
        Utils.setKwargs(self, **kwargs)

    srcPair = BaseSrc  #: Source Pair

    @property
    def srcList(self):
        """Source List"""
        return getattr(self, '_srcList', None)

    @srcList.setter
    def srcList(self, value):
        assert type(value) is list, 'srcList must be a list'
        assert np.all([isinstance(src, self.srcPair) for src in value]), (
            'All sources must be instances of {0!s}'.format(
                self.srcPair.__name__
            )
        )
        assert len(set(value)) == len(value), 'The srcList must be unique'
        self._srcList = value
        self._sourceOrder = dict()
        [
            self._sourceOrder.setdefault(src.uid, ii) for ii, src in
            enumerate(self._srcList)
        ]

    def getSourceIndex(self, sources):
        if type(sources) is not list:
            sources = [sources]
        for src in sources:
            if getattr(src, 'uid', None) is None:
                raise KeyError(
                    'Source does not have a uid: {0!s}'.format(str(src))
                )
        inds = list(map(
            lambda src: self._sourceOrder.get(src.uid, None), sources
        ))
        if None in inds:
            raise KeyError(
                'Some of the sources specified are not in this survey. '
                '{0!s}'.format(str(inds))
            )
        return inds

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
        raise Exception(
            'Pair survey to a problem to access the problems mesh.'
        )

    def pair(self, p):
        """Bind a problem to this survey instance using pointers"""
        assert hasattr(p, 'surveyPair'), (
            "Problem must have an attribute 'surveyPair'."
        )
        assert isinstance(self, p.surveyPair), (
            "Problem requires survey object must be an instance of a {0!s} "
            "class.".format((p.surveyPair.__name__))
        )
        if p.ispaired:
            raise Exception(
                "The problem object is already paired to a survey. "
                "Use prob.unpair()"
            )
        elif self.ispaired:
            raise Exception(
                "The survey object is already paired to a problem. "
                "Use survey.unpair()"
            )
        self._prob = p
        p._survey = self

    def unpair(self):
        """Unbind a problem from this survey instance"""
        if not self.ispaired: return
        self.prob._survey = None
        self._prob = None

    @property
    def ispaired(self):
        return self.prob is not None

    @property
    def nD(self):
        """Number of data"""
        return self.vnD.sum()

    @property
    def vnD(self):
        """Vector number of data"""
        return np.array([src.nD for src in self.srcList])

    @property
    def nSrc(self):
        """Number of Sources"""
        return len(self.srcList)

    @Utils.count
    @Utils.requires('prob')
    def dpred(self, m=None, f=None):
        """dpred(m, f=None)

            Create the projected data from a model.
            The fields, f, (if provided) will be used for the predicted data
            instead of recalculating the fields (which may be expensive!).

            .. math::

                d_\\text{pred} = P(f(m))

            Where P is a projection of the fields onto the data space.
        """
        if f is None:
            f = self.prob.fields(m)
        return Utils.mkvc(self.eval(f))

    @Utils.count
    def eval(self, f):
        """eval(f)

            This function projects the fields onto the data space.

            .. math::

                d_\\text{pred} = \mathbf{P} f(m)
        """
        raise NotImplementedError('eval is not yet implemented.')

    @Utils.count
    def evalDeriv(self, f):
        """evalDeriv(f)

            This function s the derivative of projects the fields onto the data space.

            .. math::

                \\frac{\partial d_\\text{pred}}{\partial u} = \mathbf{P}
        """
        raise NotImplementedError('eval is not yet implemented.')

    @Utils.count
    def residual(self, m, f=None):
        """residual(m, f=None)

            :param numpy.array m: geophysical model
            :param numpy.array f: fields
            :rtype: numpy.array
            :return: data residual

            The data residual:

            .. math::

                \mu_\\text{data} = \mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs}

        """
        return Utils.mkvc(self.dpred(m, f=f) - self.dobs)

    @property
    def isSynthetic(self):
        "Check if the data is synthetic."
        return self.mtrue is not None

    def makeSyntheticData(self, m, std=0.05, f=None, force=False):
        """
            Make synthetic data given a model, and a standard deviation.

            :param numpy.array m: geophysical model
            :param numpy.array std: standard deviation
            :param numpy.array u: fields for the given model (if pre-calculated)
            :param bool force: force overwriting of dobs

        """
        if getattr(self, 'dobs', None) is not None and not force:
            raise Exception(
                'Survey already has dobs. You can use force=True to override '
                'this exception.'
            )
        self.mtrue = m
        self.dtrue = self.dpred(m, f=f)
        noise = std*abs(self.dtrue)*np.random.randn(*self.dtrue.shape)
        self.dobs = self.dtrue+noise
        self.std = self.dobs*0 + std
        return self.dobs


class LinearSurvey(BaseSurvey):
    def eval(self, f):
        return f

    @property
    def nD(self):
        return self.prob.G.shape[0]
