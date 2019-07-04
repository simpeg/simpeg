import numpy as np
import scipy.sparse as sp
import uuid
import properties
import warnings

from .utils import mkvc, Counter, setKwargs
from .props import BaseSimPEG


class RxLocationArray(properties.Array):

    class_info = "an array of receiver locations"

    def validate(self, instance, value):
        if len(value.shape) == 1:
            value = mkvc(value, 2)
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
        "Locations of the receivers (nRx x nDim)",
        shape=("*", "*"),
        required=True
    )

    # TODO: project_grid?
    projGLoc = properties.StringChoice(
        "Projection grid location, default is CC",
        choices=["CC", "Fx", "Fy", "Fz", "Ex", "Ey", "Ez", "N"],
        default="CC"
    )

    # TODO: store_projections
    storeProjections = properties.Bool(
        "Store calls to getP (organized by mesh)",
        default=True
    )

    _uid = properties.Uuid(
        "unique ID for the receiver"
    )

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
        if getattr(self, '_Ps', None) is None:
            self._Ps = {}

    @property
    def locs(self):
        warnings.warn(
            "BaseRx.locs will be deprecaited and replaced with "
            "BaseRx.locations. Please update your code accordingly"
        )
        return self.locations

    @locs.setter
    def locs(self, value):
        warnings.warn(
            "BaseRx.locs will be deprecaited and replaced with "
            "BaseRx.locations. Please update your code accordingly"
        )
        self.locations = value

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
        "times where the recievers measure data",
        shape=("*",),
        required=True
    )

    projTLoc = properties.StringChoice(
        "location on the time mesh where the data are projected from",
        choices=["N", "CC"],
        default="N"
    )

    def __init__(self, locations=None, times=None, **kwargs):
        super(BaseTimeRx, self).__init__(locations=locations, **kwargs)
        if times is not None:
            self.times=times

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
        "receiver list", properties.Instance("a SimPEG receiver", BaseRx),
        default=[]
    )

    _uid = properties.Uuid(
        "unique identifier for the source"
    )

    @property
    def loc(self):
        warnings.warn(
            "BaseSrc.loc will be deprecaited and replaced with "
            "BaseSrc.location. Please update your code accordingly"
        )
        return self.location

    @loc.setter
    def loc(self, value):
        warnings.warn(
            "BaseSrc.loc will be deprecaited and replaced with "
            "BaseSrc.location. Please update your code accordingly"
        )
        self.location = value

    @property
    def rxList(self):
        warnings.warn(
            "source.rxList will be deprecaited and replaced with "
            "source.receiver_list. Please update your code accordingly"
        )
        return self.receiver_list

    @rxList.setter
    def rxList(self, value):
        warnings.warn(
            "source.rxList will be deprecaited and replaced with "
            "source.receiver_list. Please update your code accordingly"
        )
        self.receiver_list = value

    @properties.validator('receiver_list')
    def _receiver_list_validator(self, change):
        value = change['value']
        assert len(set(value)) == len(value), 'The receiver_list must be unique'
        self._rxOrder = dict()
        [
            self._rxOrder.setdefault(rx._uid, ii) for ii, rx in
            enumerate(value)
        ]

    def getReceiverIndex(self, receiver):
        if type(receiver) is not list:
            receiver = [receiver]
        for rx in receiver:
            if getattr(rx, '_uid', None) is None:
                raise KeyError(
                    'Source does not have a _uid: {0!s}'.format(str(rx))
                )
        inds = list(map(
            lambda rx: self._rxOrder.get(rx._uid, None), receiver
        ))
        if None in inds:
            raise KeyError(
                'Some of the receiver specified are not in this survey. '
                '{0!s}'.format(str(inds))
            )
        return inds

    @property
    def nD(self):
        """Number of data"""
        return self.vnD.sum()

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

    counter = properties.Instance(
        "A SimPEG counter object",
        Counter
    )

    source_list = properties.List(
        "A list of sources for the survey",
        properties.Instance("A SimPEG source", BaseSrc),
        default=[]
    )

    def __init__(self, source_list=None, **kwargs):
        super(BaseSurvey, self).__init__(**kwargs)
        if source_list is not None:
            self.source_list = source_list

    @properties.validator('source_list')
    def _source_list_validator(self, change):
        value = change['value']
        if len(set(value)) != len(value):
            raise Exception('The source_list must be unique')
        self._sourceOrder = dict()
        [
            self._sourceOrder.setdefault(src._uid, ii) for ii, src in
            enumerate(value)
        ]

    # TODO: this should be private
    def getSourceIndex(self, sources):
        if type(sources) is not list:
            sources = [sources]

        for src in sources:
            if getattr(src, '_uid', None) is None:
                raise KeyError(
                    'Source does not have a _uid: {0!s}'.format(str(src))
                )
        inds = list(map(
            lambda src: self._sourceOrder.get(src._uid, None), sources
        ))
        if None in inds:
            raise KeyError(
                'Some of the sources specified are not in this survey. '
                '{0!s}'.format(str(inds))
            )
        return inds

    @property
    def nD(self):
        """Number of data"""
        return self.vnD.sum()

    @property
    def vnD(self):
        """Vector number of data"""
        if getattr(self, '_vnD', None) is None:
            self._vnD = np.array([src.nD for src in self.source_list])
        return self._vnD

    @property
    def nSrc(self):
        """Number of Sources"""
        return len(self.source_list)

    #############
    # Depreciated
    #############

    def dpred(self, m=None, f=None):
        raise Exception(
            "Survey no longer has the dpred method. Please use "
            "simulation.dpred instead"
        )

    @property
    def srcList(self):
        warnings.warn(
            "srcList has been renamed to source_list. Please update your code "
            "accordingly"
        )
        return self.source_list

    @srcList.setter
    def srcList(self, value):
        warnings.warn(
            "srcList has been renamed to source_list. Please update your code "
            "accordingly"
        )
        self.source_list = value

    def pair(self, simulation):
        warnings.warn(
            "survey.pair(simulation) will be depreciated. Please update your code "
            "to instead use simulation.survey = survey"
        )
        simulation.survey = self

    def makeSyntheticData(self, m, std=None, f=None, force=False, **kwargs):
        raise Exception(
            "Survey no longer has the makeSyntheticData method. Please use "
            "simulation.make_synthetic_data instead."
        )


class BaseTimeSurvey(BaseSurvey):

    @property
    def unique_times(self):
        if getattr(self, '_unique_times', None) is None:
            rx_times = []
            for source in self.source_list:
                for receiver in source.receiver_list:
                    rx_times.append(receiver.times)
            self._unique_times = np.unique(np.hstack(rx_times))
        return self._unique_times


###############################################################################
#
# Classes to be depreciated
#
###############################################################################

class LinearSurvey:
    """
    Survey for a linear problem
    """
    def __init__(self, source_list=None, **kwargs):
        warnings.warn(
            "LinearSurvey will be depreciated. Please use survey.BaseSurvey "
            "instead", DeprecationWarning
        )
        BaseSurvey.__init__(self, source_list, **kwargs)


class Data:
    def __init__(self, survey=None, data=None, **kwargs):
        raise Exception(
            "survey.Data has been depreciated. To access the data class. To "
            "import the data class, please use SimPEG.data.Data"
        )
