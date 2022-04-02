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
    """Locations array for receivers"""

    class_info = "an array of receiver locations"

    def validate(self, instance, value):
        """Validation method for setting locations array

        Parameters
        ----------
        instance : class
            The class used to validate the input argument *value*
        value :
            The input used to define the locations for a given receiver.

        Returns
        -------
        properties.Array
            The receiver location array
        """
        value = np.atleast_2d(value)
        return super(RxLocationArray, self).validate(instance, value)


class SourceLocationArray(properties.Array):
    """Locations array for sources"""

    class_info = "a 1D array denoting the source location"

    def validate(self, instance, value):
        """Validation method for setting locations array

        Parameters
        ----------
        instance : class
            The class used to validate the input argument *value*
        value :
            The input used to define the locations for a given source.

        Returns
        -------
        properties.Array
            The source location array
        """
        if not isinstance(value, np.ndarray):
            value = np.atleast_1d(np.array(value))
        return super(SourceLocationArray, self).validate(instance, value)


class BaseRx:
    """Base SimPEG receiver class.

    Parameters
    ----------
    locations : (n_loc, ndim) numpy.ndarray
        Locations assocated with a given receiver
    projGLoc : str in ["CC", "Fx", "Fy", "Fz", "Ex", "Ey", "Ez", "N"], Default='CC'
        Sets the location of numerical solution
    storeProjections : bool, Default=``False``
        Store projections from the mesh to receiver
    uid : uuid.UUID
        A universally unique identifier
    """

    _Ps = None

    def __init__(self, locations=None, projGLoc='CC', storeProjections=False, uid=None, **kwargs):

        locs = kwargs.pop("locs", None)
        if locs is not None:
            warnings.warn(
                "'locs' is a deprecated property. Please use 'locations' instead."
                "'locs' be removed in SimPEG 0.16.0."
            )
            locations = locs
        rxType = kwargs.pop("rxType", None)
        if rxType is not None:
            warnings.warn(
                "BaseRx no longer has an rxType. Each rxType should instead "
                "be a different receiver class."
            )
        if getattr(self, "_Ps", None) is None:
            self._Ps = {}

        if locations is not None:
            self.locations = locations
        self.projGLoc = projGLoc
        self.storeProjections = storeProjections
        if uid is None:
            self.uid = uuid.uuid4()
        else:
            self.uid = uid

    @property
    def locations(self):
        """Receiver locations

        Returns
        -------
        (n_loc, n_dim) np.ndarray
            Receiver locations.
        """
        return self._locations

    @locations.setter
    def locations(self, locs):
        try:
            locs = np.atleast_2d(locs).astype(float)
        except:
            raise TypeError(f"locations must be (n_loc, n_dim) array_like, got {type(locs)}")
        self._locations = locs

    @property
    def projGLoc(self):
        """Gridded locations being projected to receiver locations.

        A ``str`` is used to the define the projection from the gridded locations
        on the mesh to the receiver locations. The choices are as follows:

        - "CC"  --> cell centers
        - "Fx"  --> x-faces
        - "Fy"  --> y-faces
        - "Fz"  --> z-faces
        - "Ex"  --> x-edges
        - "Ey"  --> y-edges
        - "Ez"  --> z-edges
        - "N"   --> nodes

        Returns
        -------
        str
            The gridded locations being projected to the receiver locations.
        """
        return self._projGLoc

    @projGLoc.setter
    def projGLoc(self, var):
        if (var in ["CC", "Fx", "Fy", "Fz", "Ex", "Ey", "Ez", "N"]) == False:
            raise TypeError(
                "projGLoc must be one of the following strings: "
                "'CC', 'Fx', 'Fy', 'Fz', 'Ex', 'Ey', 'Ez', 'N'"
                f"Got {type(locs)}"
            )

        self._projGLoc = var

    @property
    def uid(self):
        """Universal unique identifier

        Returns
        -------
        uuid.UUID
            A universal unique identifier
        """
        return self._uid

    @uid.setter
    def uid(self, var):
        if not isinstance(var, uuid.UUID):
            raise TypeError(f"uid must be an instance of uuid.UUID. Got {type(var)}")
        self._uid = var


    # TODO: write a validator that checks against mesh dimension in the
    # BaseSimulation
    # TODO: location
    # locations = RxLocationArray(
    #     "Locations of the receivers (nRx x nDim)", shape=("*", "*"), required=True
    # )

    # TODO: project_grid?
    # projGLoc = properties.StringChoice(
    #     "Projection grid location, default is CC",
    #     choices=["CC", "Fx", "Fy", "Fz", "Ex", "Ey", "Ez", "N"],
    #     default="CC",
    # )

    # TODO: store_projections
    # storeProjections = properties.Bool(
    #     "Store calls to getP (organized by mesh)", default=True
    # )

    # _uid = properties.Uuid("unique ID for the receiver")

    # _Ps = properties.Dictionary("dictonary for storing projections",)

    

    @property
    def nD(self):
        """Number of data associated with the receiver

        Returns
        -------
        int
            Number of data associated with the receiver
        """
        return self.locations.shape[0]

    def getP(self, mesh, projGLoc=None):
        """Get projection matrix from mesh to recei

        Parameters
        ----------
        mesh : discretize.BaseMesh
            A discretize mesh
        projGLoc : str
            Define what part of the mesh (i.e. edges, faces, centers, nodes) to
            project from. Must be one of::

                'Ex', 'edges_x'           -> x-component of field defined on x edges
                'Ey', 'edges_y'           -> y-component of field defined on y edges
                'Ez', 'edges_z'           -> z-component of field defined on z edges
                'Fx', 'faces_x'           -> x-component of field defined on x faces
                'Fy', 'faces_y'           -> y-component of field defined on y faces
                'Fz', 'faces_z'           -> z-component of field defined on z faces
                'N', 'nodes'              -> scalar field defined on nodes
                'CC', 'cell_centers'      -> scalar field defined on cell centers
                'CCVx', 'cell_centers_x'  -> x-component of vector field defined on cell centers
                'CCVy', 'cell_centers_y'  -> y-component of vector field defined on cell centers
                'CCVz', 'cell_centers_z'  -> z-component of vector field defined on cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            P, the interpolation matrix
        """
        if projGLoc is None:
            projGLoc = self.projGLoc

        if (mesh, projGLoc) in self._Ps:
            return self._Ps[(mesh, projGLoc)]

        P = mesh._getInterpolationMat(self.locations, projGLoc)
        if self.storeProjections:
            self._Ps[(mesh, projGLoc)] = P
        return P

    def eval(self, **kwargs):
        """Not currently implemented"""
        raise NotImplementedError(
            "the eval method for {} has not been implemented".format(self)
        )

    def evalDeriv(self, **kwargs):
        """Not currently implemented"""
        raise NotImplementedError(
            "the evalDeriv method for {} has not been implemented".format(self)
        )


class BaseTimeRx(BaseRx):
    """Base SimPEG receiver class for time-domain simulations"""

    # times = properties.Array(
    #     "times where the recievers measure data", shape=("*",), required=True
    # )

    # projTLoc = properties.StringChoice(
    #     "location on the time mesh where the data are projected from",
    #     choices=["N", "CC"],
    #     default="N",
    # )

    def __init__(self, locations=None, times=None, **kwargs):
        super(BaseTimeRx, self).__init__(locations=locations, **kwargs)
        if times is not None:
            self.times = times

    @property
    def times(self):
        """Time channels for the receiver

        Returns
        -------
        numpy.ndarray
            Time channels for the receiver
        """
        return self._times

    @times.setter
    def times(self, value):
        # Ensure float or numpy array of float
        try:
            value = np.atleast_1d(value).astype(float)
        except:
            raise TypeError(f"times is not a valid type. Got {type(value)}")
        
        if value.ndim > 1:
            raise TypeError(f"times must be ('*') array")

        self._times = value


    @property
    def projTLoc(self):
        """Define gridding for projection from all time steps to receiver time channels.

        A ``str`` is used to the define gridding of the time steps and how they are
        projected to the time channels. The choices are as follows:

        - "CC": time-steps defined as cell centers
        - "N": time-steps defined as nodes

        Returns
        -------
        str
            The gridding used for the time-steps.
        """
        return self._projTLoc

    @projTLoc.setter
    def projTLoc(self, var):
        if (var in ["CC", "N"]) == False:
            raise TypeError(
                f"projTLoc must be 'CC' or 'N'. Got {type(locs)}"
            )

        self._projTLoc = var
    

    @property
    def nD(self):
        """Number of data associated with the receiver.

        Returns
        -------
        int
            Number of data associated with the receiver
        """
        return self.locations.shape[0] * len(self.times)

    def getSpatialP(self, mesh):
        """Returns the spatial projection matrix from mesh to receivers.

        Parameters
        ----------
        mesh: discretize.BaseMesh
            A ``discretize`` mesh; i.e. ``TensorMesh``, ``CylMesh``, ``TreeMesh``

        Returns
        -------
        scipy.sparse.csr_matrix
            The projection matrix from the mesh (one of cell centers, nodes,
            edges, etc...) to the receivers. The returned quantity is not stored
            in memory. Instead, it is created on demand when needed.
        """
        return mesh.getInterpolationMat(self.locations, self.projGLoc)

    def getTimeP(self, time_mesh):
        """Returns the time projection matrix from all time steps to receiver time channels.

        Parameters
        ----------
        time_mesh: 1D discretize.TensorMesh
            A 1D tensor mesh defining the time steps; either at cell centers or nodes.

        Returns
        -------
        scipy.sparse.csr_matrix
            The projection matrix from the mesh (one of cell centers, nodes,
            edges, etc...) to the receivers. The returned quantity is not stored
            in memory. Instead, it is created on demand when needed.
        """
        return time_mesh.getInterpolationMat(self.times, self.projTLoc)

    def getP(self, mesh, timeMesh):
        """
        Returns the projection matrices as a
        list for all components collected by
        the receivers.

        Notes
        -----
        Projection matrices are stored as a dictionary (mesh, timeMesh)
        if `storeProjections` is ``True``
        """
        if (mesh, timeMesh) in self._Ps:
            return self._Ps[(mesh, timeMesh)]

        Ps = self.getSpatialP(mesh)
        Pt = self.getTimeP(timeMesh)
        P = sp.kron(Pt, Ps)

        if self.storeProjections:
            self._Ps[(mesh, timeMesh)] = P

        return P


class BaseSrc:
    """Base SimPEG source class.

    Parameters
    ----------
    location : (n_dim) numpy.ndarray
        Location of the source
    receiver_list : list of SimPEG.survey.BaseRx objects
        Sets the receivers associated with the source
    uid : uuid.UUID
        A universally unique identifier
    """

    def __init__(self, receiver_list=None, location=None, uid=None, **kwargs):

        loc = kwargs.pop("loc", None)
        if loc is not None:
            warnings.warn(
                "'loc' is a deprecated property. Please use 'location' instead."
                "'loc' be removed in SimPEG 0.16.0."
            )
            location = loc

        rxList = kwargs.pop("rxList", None)
        if rxList is not None:
            warnings.warn(
                "'rxList' is a deprecated property. Please use 'receiver_list' instead."
                "'rxList' be removed in SimPEG 0.16.0."
            )
            receiver_list = rxList

        if location is not None:
            self.location = location

        if receiver_list is not None:
            self.receiver_list = receiver_list

        if uid is None:
            self.uid = uuid.uuid4()
        else:
            self.uid = uid

    # location = SourceLocationArray(
    #     "Location of the source [x, y, z] in 3D", shape=("*",), required=False
    # )

    @property
    def location(self):
        """Source location

        Returns
        -------
        (n_dim) np.ndarray
            Source location.
        """
        return self._location

    @location.setter
    def location(self, loc):
        try:
            loc = np.atleast_1d(loc).astype(float)
        except:
            raise TypeError(f"location must be (n_dim) array_like, got {type(loc)}")

        if loc.ndim > 1:
            raise TypeError(f"location must be (n_dim) array_like, got {type(loc)}")

        self._location = loc

    # receiver_list = properties.List(
    #     "receiver list", properties.Instance("a SimPEG receiver", BaseRx), default=[]
    # )

    @property
    def receiver_list(self):
        """List of receivers associated with the source

        Returns
        -------
        list of SimPEG.survey.BaseRx
            List of receivers associated with the source
        """
        return self._receiver_list

    @receiver_list.setter
    def receiver_list(self, new_list):

        if isinstance(new_list, BaseRx):
            new_list = [new_list]
        elif isinstance(new_list, list):
            pass
        else:
            raise TypeError("Receiver list must be a list of SimPEG.survey.BaseRx")

        assert len(set(new_list)) == len(new_list), "The receiver_list must be unique. Cannot re-use receivers"

        self._rxOrder = dict()
        [self._rxOrder.setdefault(rx._uid, ii) for ii, rx in enumerate(new_list)]
        self._receiver_list = new_list

    # @properties.validator("receiver_list")
    # def _receiver_list_validator(self, change):
    #     value = change["value"]
    #     assert len(set(value)) == len(value), "The receiver_list must be unique"
    #     self._rxOrder = dict()
    #     [self._rxOrder.setdefault(rx._uid, ii) for ii, rx in enumerate(value)]


    @property
    def uid(self):
        """Universal unique identifier

        Returns
        -------
        uuid.UUID
            A universal unique identifier
        """
        return self._uid

    @uid.setter
    def uid(self, var):
        if not isinstance(var, uuid.UUID):
            raise TypeError(f"uid must be an instance of uuid.UUID. Got {type(var)}")
        self._uid = var

    # _uid = properties.Uuid("unique identifier for the source")

    loc = deprecate_property(
        location, "loc", new_name="location", removal_version="0.16.0", error=True
    )

    rxList = deprecate_property(
        receiver_list,
        "rxList",
        new_name="receiver_list",
        removal_version="0.16.0",
        error=True,
    )

    def get_receiver_indices(self, receivers):
        """Get indices for a subset of receivers within the source's receivers list. 

        Parameters
        ----------
        receivers : list of SimPEG.survey.BaseRx
            A subset list of receivers within the source's receivers list

        Returns
        -------
        np.ndarray of int
            Indices for the subset receivers 
        """

        if not isinstance(receivers, list):
            receivers = [receivers]
        for rx in receivers:
            if getattr(rx, "_uid", None) is None:
                raise KeyError("Receiver does not have a _uid: {0!s}".format(str(rx)))
        inds = list(map(lambda rx: self._rxOrder.get(rx._uid, None), receivers))
        if None in inds:
            raise KeyError(
                "Some of the receiver specified are not in this survey. "
                "{0!s}".format(str(inds))
            )
        return inds

    @property
    def nD(self):
        """Number of data associated with the source.

        Returns
        -------
        int
            Total number of data associated with the source.
        """
        return self.vnD.sum()

    @property
    def vnD(self):
        """Vector number of data.

        Returns
        -------
        np.ndarray of int
            Returns the corresponding number of data for each receiver.
        """
        return np.array([rx.nD for rx in self.receiver_list])

    getReceiverIndex = deprecate_method(
        get_receiver_indices,
        "getReceiverIndex",
        future_warn=True,
        removal_version="0.16.0"
    )


# TODO: allow a reciever list to be provided and assume it is used for all
# sources? (and store the projections)
class BaseSurvey:
    """Base SimPEG survey class.

    Parameters
    ----------
    source_list : list of SimPEG.survey.BaseSrc objects
        Sets the sources (and their receivers)
    uid : uuid.UUID
        A universally unique identifier
    """

    def __init__(self, source_list=None, uid=None, counter=None, **kwargs):

        srcList = kwargs.pop("srcList", None)
        if srcList is not None:
            warnings.warn(
                "'srcList' is a deprecated property. Please use 'source_list' instead."
                "'srcList' be removed in SimPEG 0.16.0."
            )
            source_list = srcList

        if source_list is not None:
            self.source_list = source_list

        if uid is None:
            self.uid = uuid.uuid4()
        else:
            self.uid = uid

        if counter is not None:
            self.counter = counter


    @property
    def source_list(self):
        """List of sources associated with the survey

        Returns
        -------
        list of SimPEG.survey.BaseSrc
            List of sources associated with the survey
        """
        return self._source_list

    @source_list.setter
    def source_list(self, new_list):

        if isinstance(new_list, BaseSrc):
            new_list = [new_list]
        elif isinstance(new_list, list):
            pass
        else:
            raise TypeError("Source list must be a list of SimPEG.survey.BaseSrc")

        assert len(set(new_list)) == len(new_list), "The source_list must be unique. Cannot re-use sources"

        self._sourceOrder = dict()
        [self._sourceOrder.setdefault(src._uid, ii) for ii, src in enumerate(new_list)]
        self._source_list = new_list



    @property
    def counter(self):
        """A SimPEG counter object for counting iterations and operations

        Returns
        -------
        SimPEG.utils.counter_utils.Counter
            A SimPEG counter object
        """
        return self._counter

    @counter.setter
    def counter(self, new_obj):

        if not isinstance(new_obj, Counter):
            TypeError(f"Must be a SimPEG counter object. Got {type(new_obj)}")

        self._counter = new_obj

    # source_list = properties.List(
    #     "A list of sources for the survey",
    #     properties.Instance("A SimPEG source", BaseSrc),
    #     default=[],
    # )

    # def __init__(self, source_list=None, **kwargs):
    #     super(BaseSurvey, self).__init__(**kwargs)
    #     if source_list is not None:
    #         self.source_list = source_list

    # @properties.validator("source_list")
    # def _source_list_validator(self, change):
    #     value = change["value"]
    #     if len(set(value)) != len(value):
    #         raise Exception("The source_list must be unique")
    #     self._sourceOrder = dict()
    #     [self._sourceOrder.setdefault(src._uid, ii) for ii, src in enumerate(value)]

    # TODO: this should be private
    def get_source_indices(self, sources):
        if not isinstance(sources, list):
            sources = [sources]

        for src in sources:
            if getattr(src, "_uid", None) is None:
                raise KeyError("Source does not have a _uid: {0!s}".format(str(src)))
        inds = list(map(lambda src: self._sourceOrder.get(src._uid, None), sources))
        if None in inds:
            raise KeyError(
                "Some of the sources specified are not in this survey. "
                "{0!s}".format(str(inds))
            )
        return inds

    @property
    def nD(self):
        """Total number of data for the survey

        Returns
        -------
        int
            Total number of data for the survey
        """
        return self.vnD.sum()

    @property
    def vnD(self):
        """Number of associated data for each source

        Returns
        -------
        (n_src) np.ndarray of int
            Number of associate data for each source
        """
        if getattr(self, "_vnD", None) is None:
            self._vnD = np.array([src.nD for src in self.source_list])
        return self._vnD

    @property
    def nSrc(self):
        """Number of Sources

        Returns
        -------
        int
            Number of sources
        """
        return len(self.source_list)

    #############
    # Deprecated
    #############
    srcList = deprecate_property(
        source_list,
        "srcList",
        new_name="source_list",
        removal_version="0.16.0",
        error=True,
    )

    getSourceIndex = deprecate_method(
        get_source_indices,
        "getSourceIndex",
        future_warn=True,
        removal_version="0.16.0"
    )

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
        raise TypeError(
            "survey.pair(simulation) will be removed. Please update your code "
            "to instead use simulation.survey = survey, or pass it upon intialization "
            "of the simulation object."
        )

class BaseTimeSurvey(BaseSurvey):
    """Base SimPEG survey class for time-dependent simulations."""

    def __init__(self, source_list=None, uid=None, counter=None, **kwargs):
        super(BaseTimeSurvey, self).__init__(
            source_list=source_list, uid=uid, counter=counter, **kwargs
        )


    @property
    def unique_times(self):
        """Unique time channels for all survey receivers.

        Returns
        -------
        np.ndarray
            The unique time channels for all survey receivers.
        """
        if getattr(self, "_unique_times", None) is None:
            rx_times = []
            for source in self.source_list:
                for receiver in source.receiver_list:
                    rx_times.append(receiver.times)
            self._unique_times = np.unique(np.hstack(rx_times))
        return self._unique_times

    times = deprecate_property(
        unique_times,
        "times",
        new_name="unique_times",
        removal_version="0.16.0",
        error=True,
    )


###############################################################################
#
# Classes to be depreciated
#
###############################################################################


@deprecate_class(removal_version="0.16.0", error=True)
class LinearSurvey(BaseSurvey):
    pass


#  The data module will add this to survey when SimPEG is initialized.
# class Data:
#     def __init__(self, survey=None, data=None, **kwargs):
#         raise Exception(
#             "survey.Data has been moved. To import the data class"
#             "please use SimPEG.data.Data"
#         )
