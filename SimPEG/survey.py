import numpy as np
import scipy.sparse as sp
import uuid
import properties
import warnings
from .utils.code_utils import deprecate_property, deprecate_class, deprecate_method

from .utils import mkvc, Counter
from .props import BaseSimPEG
import types


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

    locs = deprecate_property(locations, 'locs', removal_version='0.15.0')

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
        "receiver list", properties.Instance("a SimPEG receiver", BaseRx),
        default=[]
    )

    _uid = properties.Uuid(
        "unique identifier for the source"
    )

    loc = deprecate_property(location, 'loc', removal_version='0.15.0')

    @properties.validator('receiver_list')
    def _receiver_list_validator(self, change):
        value = change['value']
        assert len(set(value)) == len(value), 'The receiver_list must be unique'
        self._rxOrder = dict()
        [
            self._rxOrder.setdefault(rx._uid, ii) for ii, rx in
            enumerate(value)
        ]

    rxList = deprecate_property(receiver_list, 'rxList', removal_version='0.15.0')

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
    # Deprecated
    #############
    srcList = deprecate_property(source_list, 'srcList', removal_version='0.15.0')

    def dpred(self, m=None, f=None):
        raise Exception(
            "Survey no longer has the dpred method. Please use "
            "simulation.dpred instead"
        )

    def makeSyntheticData(self, m, std=None, f=None, force=False, **kwargs):
        raise Exception(
            "Survey no longer has the makeSyntheticData method. Please use "
            "simulation.make_synthetic_data instead."
        )

    def pair(self, simulation):
        warnings.warn(
            "survey.pair(simulation) will be deprecated. Please update your code "
            "to instead use simulation.survey = survey, or pass it upon intialization "
            "of the simulation object. This will be removed in version "
            "0.15.0 of SimPEG", DeprecationWarning
        )
        simulation.survey = self
        self.simulation = simulation

        def dep_dpred(target, m=None, f=None):
            warnings.warn(
                "The Survey.dpred method has been deprecated. Please use "
                "simulation.dpred instead. This will be removed in version "
                "0.15.0 of SimPEG", DeprecationWarning
            )
            return target.simulation.dpred(m=m, f=f)
        self.dpred = types.MethodType(dep_dpred, self)

        def dep_makeSyntheticData(target, m, std=None, f=None, **kwargs):
            warnings.warn(
                "The Survey.makeSyntheticData method has been deprecated. Please use "
                "simulation.make_synthetic_data instead. This will be removed in version "
                "0.15.0 of SimPEG", DeprecationWarning
            )
            if std is None and getattr(target, 'std', None) is None:
                stddev = 0.05
                print(
                        'SimPEG.Survey assigned default std '
                        'of 5%'
                    )
            elif std is None:
                stddev = target.std
            else:
                stddev = std
                print(
                        'SimPEG.Survey assigned new std '
                        'of {:.2f}%'.format(100.*stddev)
                    )

            data = target.simulation.make_synthetic_data(
                m, standard_deviation=stddev, f=f, add_noise=True)
            target.dtrue = data.dclean
            target.dobs = data.dobs
            target.std = data.standard_deviation
            return target.dobs
        self.makeSyntheticData = types.MethodType(dep_makeSyntheticData, self)

        def dep_residual(target, m, f=None):
            warnings.warn(
                "The Survey.residual method has been deprecated. Please use "
                "L2DataMisfit.residual instead. This will be removed in version "
                "0.15.0 of SimPEG", DeprecationWarning
            )
            return mkvc(target.dpred(m, f=f) - target.dobs)
        self.residual = types.MethodType(dep_residual, self)


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

    times = deprecate_property(unique_times, 'times', removal_version='0.15.0')


###############################################################################
#
# Classes to be depreciated
#
###############################################################################

@deprecate_class(removal_version='0.15.0')
class LinearSurvey(BaseSurvey):
    pass

#  The data module will add this to survey when SimPEG is initialized.
# class Data:
#     def __init__(self, survey=None, data=None, **kwargs):
#         raise Exception(
#             "survey.Data has been moved. To import the data class"
#             "please use SimPEG.data.Data"
#         )
