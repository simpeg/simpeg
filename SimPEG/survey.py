import numpy as np
import scipy.sparse as sp

# import properties
import warnings
import uuid

from .utils import Counter
from .utils.code_utils import (
    deprecate_property,
    deprecate_method,
    validate_location_property,
    validate_ndarray_with_shape,
    validate_list_of_types,
    validate_type,
)
from .props import BaseSimPEG

# class RxLocationArray(properties.Array):
#     """Locations array for receivers"""

#     class_info = "an array of receiver locations"

#     def validate(self, instance, value):
#         """Validation method for setting locations array

#         Parameters
#         ----------
#         instance : class
#             The class used to validate the input argument *value*
#         value :
#             The input used to define the locations for a given receiver.

#         Returns
#         -------
#         properties.Array
#             The receiver location array
#         """
#         value = np.atleast_2d(value)
#         return super(RxLocationArray, self).validate(instance, value)


# class SourceLocationArray(properties.Array):
#     """Locations array for sources"""

#     class_info = "a 1D array denoting the source location"

#     def validate(self, instance, value):
#         """Validation method for setting locations array

#         Parameters
#         ----------
#         instance : class
#             The class used to validate the input argument *value*
#         value :
#             The input used to define the locations for a given source.

#         Returns
#         -------
#         properties.Array
#             The source location array
#         """
#         if not isinstance(value, np.ndarray):
#             value = np.atleast_1d(np.array(value))
#         return super(SourceLocationArray, self).validate(instance, value)


class BaseRx:
    """Base SimPEG receiver class.

    Parameters
    ----------
    locations : (n_loc, ndim) numpy.ndarray
        Locations assocated with a given receiver
    storeProjections : bool, Default = ``False``
        Store projections from the mesh to receiver
    """

    _Ps = None

    def __init__(self, locations, storeProjections=False, **kwargs):

        self.locations = locations

        # Deprecated properties
        rxType = kwargs.pop("rxType", None)
        if rxType is not None:
            raise AttributeError(
                "BaseRx no longer has an rxType property. Each receiver type is defined by "
                "a different receiver class."
            )
        projGLoc = kwargs.pop("projGLoc", None)
        if projGLoc is not None:
            warnings.warn(
                "'projGLoc' is no longer of property of the receiver class. It is set automatically "
                "based on the receiver and simulation class. Will be remove in SimPEG 0.18.0"
            )
        # ideally there shouldn't be any kwargs left to hit, but this will throw an
        # error if kwargs hasn't been emptied properly. This also will allow proper
        # initialization in multiple inheritence.
        super().__init__(**kwargs)

        # Remaining properties
        if getattr(self, "_Ps", None) is None:
            self._Ps = {}
        self.storeProjections = storeProjections
        self._uid = uuid.uuid4()

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
        self._locations = validate_ndarray_with_shape(
            "locations", locs, shape=("*", "*"), dtype=float
        )

    # @property
    # def projected_grid(self):
    #     """Gridded locations being projected to receiver locations.

    #     A ``str`` is used to the define the projection from the gridded locations
    #     on the mesh to the receiver locations. The choices are as follows:

    #     - "CC"  --> cell centers
    #     - "Fx"  --> x-faces
    #     - "Fy"  --> y-faces
    #     - "Fz"  --> z-faces
    #     - "Ex"  --> x-edges
    #     - "Ey"  --> y-edges
    #     - "Ez"  --> z-edges
    #     - "N"   --> nodes

    #     Returns
    #     -------
    #     str
    #         The gridded locations being projected to the receiver locations.
    #     """
    #     return self._projected_grid

    # @projected_grid.setter
    # def projected_grid(self, var):
    #     if (var in ["CC", "Fx", "Fy", "Fz", "Ex", "Ey", "Ez", "N"]) == False:
    #         raise TypeError(
    #             "projected_grid must be one of the following strings: "
    #             "'CC', 'Fx', 'Fy', 'Fz', 'Ex', 'Ey', 'Ez', 'N'"
    #             f"Got {type(var)}"
    #         )

    #     self._projected_grid = var

    @property
    def uid(self):
        """Universal unique identifier

        Returns
        -------
        uuid.UUID
            A universal unique identifier
        """
        return self._uid

    # TODO: write a validator that checks against mesh dimension in the
    # BaseSimulation
    # TODO: location
    # locations = RxLocationArray(
    #     "Locations of the receivers (nRx x nDim)", shape=("*", "*"), required=True
    # )

    # TODO: project_grid?
    # projected_grid = properties.StringChoice(
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

    def getP(self, mesh, projected_grid="CC"):
        """Get projection matrix from mesh to receivers

        Parameters
        ----------
        mesh : discretize.BaseMesh
            A discretize mesh
        projected_grid : str
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
        if (mesh, projected_grid) in self._Ps:
            return self._Ps[(mesh, projected_grid)]

        P = mesh.get_interpolation_matrix(self.locations, projected_grid)
        if self.storeProjections:
            self._Ps[(mesh, projected_grid)] = P
        return P

    def eval(self, **kwargs):
        """Not implemented for BaseRx"""
        raise NotImplementedError(
            "the eval method for {} has not been implemented".format(self)
        )

    def evalDeriv(self, **kwargs):
        """Not implemented for BaseRx"""
        raise NotImplementedError(
            "the evalDeriv method for {} has not been implemented".format(self)
        )


class BaseTimeRx(BaseRx):
    """Base SimPEG receiver class for time-domain simulations

    Parameters
    ----------
    locations : (n, dim) array_like
        Receiver locations
    times : array_like
        Time channels
    """

    def __init__(self, locations, times, **kwargs):
        super().__init__(locations=locations, **kwargs)

        self.times = times

    # times = properties.Array(
    #     "times where the recievers measure data", shape=("*",), required=True
    # )

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
        self._times = validate_ndarray_with_shape(
            "times", value, shape=("*",), dtype=float
        )

    # projected_time_grid = properties.StringChoice(
    #     "location on the time mesh where the data are projected from",
    #     choices=["N", "CC"],
    #     default="N",
    # )

    # @property
    # def projected_time_grid(self):
    #     """Define gridding for projection from all time steps to receiver time channels.

    #     A ``str`` is used to the define gridding of the time steps and how they are
    #     projected to the time channels. The choices are as follows:

    #     - "CC": time-steps defined as cell centers
    #     - "N": time-steps defined as nodes

    #     Returns
    #     -------
    #     str
    #         The gridding used for the time-steps.
    #     """
    #     return self._projected_time_grid

    # @projected_time_grid.setter
    # def projected_time_grid(self, var):
    #     if (var in ["CC", "N"]) == False:
    #         raise TypeError(
    #             f"projected_time_grid must be 'CC' or 'N'. Got {type(var)}"
    #         )

    #     self._projected_time_grid = var

    @property
    def nD(self):
        """Number of data associated with the receiver.

        Returns
        -------
        int
            Number of data associated with the receiver
        """
        return self.locations.shape[0] * len(self.times)

    def getSpatialP(self, mesh, projected_grid="CC"):
        """Returns the spatial projection matrix from mesh to receivers.

        Parameters
        ----------
        mesh : discretize.BaseMesh
            A ``discretize`` mesh; i.e. ``TensorMesh``, ``CylMesh``, ``TreeMesh``
        projected_grid :

        Returns
        -------
        scipy.sparse.csr_matrix
            The projection matrix from the mesh (one of cell centers, nodes,
            edges, etc...) to the receivers. The returned quantity is not stored
            in memory. Instead, it is created on demand when needed.
        """
        return mesh.get_interpolation_matrix(self.locations, projected_grid)

    def getTimeP(self, time_mesh, projected_time_grid="N"):
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
        return time_mesh.get_interpolation_matrix(self.times, projected_time_grid)

    def getP(self, mesh, time_mesh, projected_grid="CC", projected_time_grid="N"):
        """
        Returns the projection matrices as a
        list for all components collected by
        the receivers.

        Notes
        -----
        Projection matrices are stored as a dictionary (mesh, timeMesh)
        if `storeProjections` is ``True``
        """
        if (mesh, time_mesh) in self._Ps:
            return self._Ps[(mesh, time_mesh)]

        Ps = self.getSpatialP(mesh, projected_grid)
        Pt = self.getTimeP(time_mesh, projected_time_grid)
        P = sp.kron(Pt, Ps)

        if self.storeProjections:
            self._Ps[(mesh, time_mesh)] = P

        return P


class BaseSrc:
    """Base SimPEG source class.

    Parameters
    ----------
    receiver_list : list of SimPEG.survey.BaseRx objects
        Sets the receivers associated with the source
    location : (n_dim) numpy.ndarray
        Location of the source
    """

    def __init__(self, receiver_list=None, location=None, **kwargs):

        if receiver_list is None:
            receiver_list = []
        self.receiver_list = receiver_list

        if location is not None:
            self.location = location

        self._uid = uuid.uuid4()

    # add a listener to receiver_list to reset the __rxOrder.
    def __setattr__(self, name, item):
        super().__setattr__(name, item)
        if name == "receiver_list":
            self.__rxOrder = None

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
        self._location = validate_location_property("location", loc)

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
        self._receiver_list = validate_list_of_types(
            "receiver_list", new_list, BaseRx, ensure_unique=True
        )

    @property
    def _rxOrder(self):
        if getattr(self.__rxOrder, None) is None:
            self.__rxOrder = {}
            [
                self.__rxOrder.setdefault(rx.uid, ii)
                for ii, rx in enumerate(self.receiver_list)
            ]
        return self.__rxOrder

    @property
    def uid(self):
        """Universal unique identifier

        Returns
        -------
        uuid.UUID
            A universal unique identifier
        """
        return self._uid

    # _uid = properties.Uuid("unique identifier for the source")

    _fields_per_source = 1

    def get_receiver_indices(self, receivers):
        """Get indices for a subset of receivers within the source's receivers list.

        Parameters
        ----------
        receivers : list of SimPEG.survey.BaseRx
            A subset list of receivers within the source's receivers list

        Returns
        -------
        numpy.ndarray of int
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
        return sum(self.vnD)

    @property
    def vnD(self):
        """Vector number of data.

        Returns
        -------
        np.ndarray of int
            Returns the corresponding number of data for each receiver.
        """
        return np.array([rx.nD for rx in self.receiver_list])


# TODO: allow a reciever list to be provided and assume it is used for all
# sources? (and store the projections)
class BaseSurvey:
    """Base SimPEG survey class.

    Parameters
    ----------
    source_list : list of SimPEG.survey.BaseSrc objects
        Sets the sources (and their receivers)
    counter : SimPEG.utils.Counter
        A SimPEG counter object
    """

    def __init__(self, source_list, counter=None, **kwargs):

        # Source list
        if source_list is None:
            source_list = []
        self.source_list = source_list

        if counter is not None:
            self.counter = counter

        self._uid = uuid.uuid4()
        super().__init__(**kwargs)

    @property
    def source_list(self):
        """List of sources associated with the survey

        Returns
        -------
        list of SimPEG.survey.BaseSrc
            List of sources associated with the survey
        """
        return self._source_list

    # add a listener to source_list to reset the __sourceOrder.
    def __setattr__(self, name, item):
        super().__setattr__(name, item)
        if name == "source_list":
            self.__sourceOrder = None

    @source_list.setter
    def source_list(self, new_list):
        self._source_list = validate_list_of_types(
            "source_list", new_list, BaseSrc, ensure_unique=True
        )

    @property
    def _sourceOrder(self):
        if getattr(self, "__sourceOrder", None) is None:
            self.__sourceOrder = {}
            ii = 0
            for src in self.source_list:
                n_fields = src._fields_per_source
                self.__sourceOrder[src._uid] = [ii + i for i in range(n_fields)]
                ii += n_fields
        return self.__sourceOrder

    @property
    def uid(self):
        """Universal unique identifier

        Returns
        -------
        uuid.UUID
            A universal unique identifier
        """
        return self._uid

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
        self._counter = validate_type("counter", new_obj, Counter, cast=False)

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
    #     ii = 0
    #     for src in value:
    #         n_fields = src._fields_per_source
    #         self._sourceOrder[src._uid] = [ii + i for i in range(n_fields)]
    #         ii += n_fields

    # TODO: this should be private
    def get_source_indices(self, sources):
        if not isinstance(sources, list):
            sources = [sources]

        inds = []
        for src in sources:
            if getattr(src, "uid", None) is None:
                raise KeyError("Source does not have a _uid: {0!s}".format(str(src)))
            ind = self._sourceOrder.get(src.uid, None)
            if ind is None:
                raise KeyError(
                    "Some of the sources specified are not in this survey. "
                    "{0!s}".format(str(inds))
                )
            inds.extend(ind)
        return inds

    @property
    def nD(self):
        """Total number of data for the survey

        Returns
        -------
        int
            Total number of data for the survey
        """
        return sum(self.vnD)

    @property
    def vnD(self):
        """Number of associated data for each source

        Returns
        -------
        (n_src) numpy.ndarray of int
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

    @property
    def _n_fields(self):
        """number of fields required for solution"""
        return sum(src._fields_per_source for src in self.source_list)


class BaseTimeSurvey(BaseSurvey):
    """Base SimPEG survey class for time-dependent simulations."""

    @property
    def unique_times(self):
        """Unique time channels for all survey receivers.

        Returns
        -------
        numpy.ndarray
            The unique time channels for all survey receivers.
        """
        if getattr(self, "_unique_times", None) is None:
            rx_times = []
            for source in self.source_list:
                for receiver in source.receiver_list:
                    rx_times.append(receiver.times)
            self._unique_times = np.unique(np.hstack(rx_times))
        return self._unique_times
