import numpy as np
import scipy.sparse as sp
import properties
import warnings

from .utils import Counter
from .props import BaseSimPEG


class RxLocationArray(properties.Array):

    class_info = "an array of receiver locations"

    def validate(self, instance, value):
        value = np.atleast_2d(value)
        return super(RxLocationArray, self).validate(instance, value)


class SourceLocationArray(properties.Array):

    class_info = "a 1D array denoting the source location"

    def validate(self, instance, value):
        if not isinstance(value, np.ndarray):
            value = np.atleast_1d(np.array(value))
        return super(SourceLocationArray, self).validate(instance, value)


class BaseRx(properties.HasProperties):
    """SimPEG Receiver Object"""

    # TODO: write a validator that checks against mesh dimension in the
    # BaseSimulation
    # TODO: location
    locations = RxLocationArray(
        "Locations of the receivers (nRx x nDim)", shape=("*", "*"), required=True
    )

    # TODO: project_grid?
    projGLoc = properties.StringChoice(
        "Projection grid location, default is CC",
        choices=["CC", "Fx", "Fy", "Fz", "Ex", "Ey", "Ez", "N"],
        default="CC",
    )

    # TODO: store_projections
    storeProjections = properties.Bool(
        "Store calls to getP (organized by mesh)", default=True
    )

    _uid = properties.Uuid("unique ID for the receiver")

    _Ps = properties.Dictionary(
        "dictonary for storing projections",
    )

    def __init__(self, locations=None, **kwargs):
        super(BaseRx, self).__init__(**kwargs)
        if locations is not None:
            self.locations = locations
        rxType = kwargs.pop("rxType", None)
        if rxType is not None:
            warnings.warn(
                "BaseRx no longer has an rxType. Each rxType should instead "
                "be a different receiver class."
            )
        if getattr(self, "_Ps", None) is None:
            self._Ps = {}

    @property
    def nD(self):
        """Number of data in the receiver."""
        return self.locations.shape[0]

    def getP(self, mesh, projGLoc=None):
        """
        Returns the projection matrices as a
        list for all components collected by
        the receivers.

        .. note::

            Projection matrices are stored as a dictionary listed by meshes.
        """
        if projGLoc is None:
            projGLoc = self.projGLoc

        if (mesh, projGLoc) in self._Ps:
            return self._Ps[(mesh, projGLoc)]

        P = mesh.getInterpolationMat(self.locations, projGLoc)
        if self.storeProjections:
            self._Ps[(mesh, projGLoc)] = P
        return P

    def eval(self, **kwargs):
        raise NotImplementedError(
            "the eval method for {} has not been implemented".format(self)
        )

    def evalDeriv(self, **kwargs):
        raise NotImplementedError(
            "the evalDeriv method for {} has not been implemented".format(self)
        )


class BaseTimeRx(BaseRx):
    """SimPEG Receiver Object for time-domain simulations"""

    times = properties.Array(
        "times where the recievers measure data", shape=("*",), required=True
    )

    projTLoc = properties.StringChoice(
        "location on the time mesh where the data are projected from",
        choices=["N", "CC"],
        default="N",
    )

    def __init__(self, locations=None, times=None, **kwargs):
        super(BaseTimeRx, self).__init__(locations=locations, **kwargs)
        if times is not None:
            self.times = times

    @property
    def nD(self):
        """Number of data in the receiver."""
        return self.locations.shape[0] * len(self.times)

    def getSpatialP(self, mesh):
        """
        Returns the spatial projection matrix.

        .. note::

            This is not stored in memory, but is created on demand.
        """
        return mesh.getInterpolationMat(self.locations, self.projGLoc)

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


class BaseSrc(BaseSimPEG):
    """SimPEG Source Object"""

    location = SourceLocationArray(
        "Location of the source [x, y, z] in 3D", shape=("*",), required=False
    )

    receiver_list = properties.List(
        "receiver list", properties.Instance("a SimPEG receiver", BaseRx), default=[]
    )

    _uid = properties.Uuid("unique identifier for the source")

    _fields_per_source = 1

    @properties.validator("receiver_list")
    def _receiver_list_validator(self, change):
        value = change["value"]
        assert len(set(value)) == len(value), "The receiver_list must be unique"
        self._rxOrder = dict()

    def getReceiverIndex(self, receiver):
        if not isinstance(receiver, list):
            receiver = [receiver]
        for rx in receiver:
            if getattr(rx, "_uid", None) is None:
                raise KeyError("Source does not have a _uid: {0!s}".format(str(rx)))
        inds = list(map(lambda rx: self._rxOrder.get(rx._uid, None), receiver))
        if None in inds:
            raise KeyError(
                "Some of the receiver specified are not in this survey. "
                "{0!s}".format(str(inds))
            )
        return inds

    @property
    def nD(self):
        """Number of data"""
        return sum(self.vnD)

    @property
    def vnD(self):
        """Vector number of data"""
        return np.array([rx.nD for rx in self.receiver_list])

    def __init__(self, receiver_list=None, location=None, **kwargs):
        super(BaseSrc, self).__init__(**kwargs)
        if receiver_list is not None:
            self.receiver_list = receiver_list
        if location is not None:
            self.location = location


# TODO: allow a reciever list to be provided and assume it is used for all
# sources? (and store the projections)
class BaseSurvey(properties.HasProperties):
    """
    Survey holds the sources and receivers for a survey
    """

    counter = properties.Instance("A SimPEG counter object", Counter)

    source_list = properties.List(
        "A list of sources for the survey",
        properties.Instance("A SimPEG source", BaseSrc),
        default=[],
    )

    def __init__(self, source_list=None, **kwargs):
        super(BaseSurvey, self).__init__(**kwargs)
        if source_list is not None:
            self.source_list = source_list

    @properties.validator("source_list")
    def _source_list_validator(self, change):
        value = change["value"]
        if len(set(value)) != len(value):
            raise Exception("The source_list must be unique")
        self._sourceOrder = dict()
        ii = 0
        for src in value:
            n_fields = src._fields_per_source
            self._sourceOrder[src._uid] = [ii + i for i in range(n_fields)]
            ii += n_fields

    # TODO: this should be private
    def getSourceIndex(self, sources):
        if not isinstance(sources, list):
            sources = [sources]

        inds = []
        for src in sources:
            if getattr(src, "_uid", None) is None:
                raise KeyError("Source does not have a _uid: {0!s}".format(str(src)))
            ind = self._sourceOrder.get(src._uid, None)
            if ind is None:
                raise KeyError(
                    "Some of the sources specified are not in this survey. "
                    "{0!s}".format(str(inds))
                )
            inds.extend(ind)
        return inds

    @property
    def nD(self):
        """Number of data"""
        return sum(self.vnD)

    @property
    def vnD(self):
        """Vector number of data"""
        if getattr(self, "_vnD", None) is None:
            self._vnD = np.array([src.nD for src in self.source_list])
        return self._vnD

    @property
    def nSrc(self):
        """Number of Sources"""
        return len(self.source_list)

    @property
    def _n_fields(self):
        """number of fields required for solution"""
        return sum(src._fields_per_source for src in self.source_list)


class BaseTimeSurvey(BaseSurvey):
    @property
    def unique_times(self):
        if getattr(self, "_unique_times", None) is None:
            rx_times = []
            for source in self.source_list:
                for receiver in source.receiver_list:
                    rx_times.append(receiver.times)
            self._unique_times = np.unique(np.hstack(rx_times))
        return self._unique_times
