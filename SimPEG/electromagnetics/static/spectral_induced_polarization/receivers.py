import numpy as np
import properties
from ....utils.code_utils import deprecate_property
from ....utils import sdiag

from ....survey import BaseTimeRx, RxLocationArray
import warnings


class BaseRx(BaseTimeRx):

    orientation = properties.StringChoice(
        "orientation of the receiver. Must currently be 'x', 'y', 'z'", ["x", "y", "z"]
    )

    projField = properties.StringChoice(
        "field to be projected in the calculation of the data",
        choices=["phi", "e", "j"],
        default="phi",
    )

    data_type = properties.StringChoice(
        "Type of DC-IP survey",
        required=True,
        default="volt",
        choices=["volt", "apparent_resistivity", "apparent_chargeability"],
    )

    # @property
    # def projField(self):
    #     """Field Type projection (e.g. e b ...)"""
    #     return self.knownRxTypes[self.rxType][0]

    @property
    def dc_voltage(self):
        # todo : this is sketchy
        return self._dc_voltage

    def projGLoc(self, f):
        """Grid Location projection (e.g. Ex Fy ...)"""
        if self.orientation is not None:
            return f._GLoc(self.projField) + self.orientation
        return f._GLoc(self.projField)

    def getTimeP(self, timesall):
        """
            Returns the time projection matrix.

            .. note::

                This is not stored in memory, but is created on demand.
        """
        time_inds = np.in1d(timesall, self.times)
        return time_inds

    def eval(self, src, mesh, f):
        P = self.getP(mesh, self.projGLoc(f))
        proj_f = self.projField
        if proj_f == "phi":
            proj_f = "phiSolution"
        return P * f[src, proj_f]

    def evalDeriv(self, src, mesh, f, v, adjoint=False):
        P = self.getP(mesh, self.projGLoc(f))
        if not adjoint:
            return P * v
        elif adjoint:
            return P.T * v


class Dipole(BaseRx):
    """
    Dipole receiver
    """

    locations = properties.List(
        "list of locations of each electrode in a dipole receiver",
        RxLocationArray("location of electrode", shape=("*", "*")),
        min_length=1,
        max_length=2,
    )

    def __init__(
        self, locations_m=None, locations_n=None, times=None, locations=None, **kwargs
    ):
        # Check for old keywords
        if "locationsM" in kwargs.keys():
            locations_m = kwargs.pop("locationsM")
            warnings.warn(
                "The locationsM property has been deprecated. Please set the "
                "locations_m property instead. This will be removed in version"
                " 0.15.0 of SimPEG",
                DeprecationWarning,
            )

        if "locationsN" in kwargs.keys():
            locations_n = kwargs.pop("locationsN")
            warnings.warn(
                "The locationsN property has been deprecated. Please set the "
                "locations_n property instead. This will be removed in version"
                " 0.15.0 of SimPEG",
                DeprecationWarning,
            )

        # if locations_m set, then use locations_m, locations_n
        if locations_m is not None:
            if locations_n is None:
                raise ValueError(
                    "For a dipole source both locations_m and locations_n "
                    "must be set"
                )

            if locations is not None:
                raise ValueError(
                    "Cannot set both locations and locations_m, locations_n. "
                    "Please provide either locations=(locations_m, locations_n) "
                    "or both locations_m=locations_m, locations_n=locations_n"
                )

            locations = [np.atleast_2d(locations_m), np.atleast_2d(locations_n)]

        elif locations is not None:
            if len(locations) != 2:
                raise ValueError(
                    "locations must be a list or tuple of length 2: "
                    "[locations_m, locations_n]. The input locations has "
                    f"length {len(locations)}"
                )
            locations = [np.atleast_2d(locations[0]), np.atleast_2d(locations[1])]

        # check the size of locations_m, locations_n
        if locations[0].shape != locations[1].shape:
            raise ValueError(
                f"locations_m (shape: {locations[0].shape}) and "
                f"locations_n (shape: {locations[1].shape}) need to be "
                f"the same size"
            )

        super().__init__(times=times, **kwargs)
        self.locations = locations

    @property
    def locations_m(self):
        """Locations of the M-electrodes"""
        return self.locations[0]

    @property
    def locations_n(self):
        """Locations of the N-electrodes"""
        return self.locations[1]

    # this should probably be updated to n_receivers...
    @property
    def nD(self):
        """Number of data in the receiver."""
        return self.locations[0].shape[0]

    nRx = deprecate_property(nD, "nRx", new_name="nD", removal_version="0.15.0")

    def getP(self, mesh, Gloc):
        if mesh in self._Ps:
            return self._Ps[mesh]

        P0 = mesh.getInterpolationMat(self.locations[0], Gloc)
        P1 = mesh.getInterpolationMat(self.locations[1], Gloc)
        P = P0 - P1

        if self.data_type == "apparent_resistivity":
            P = sdiag(1.0 / self.geometric_factor) * P
        elif self.data_type == "apparent_chargeability":
            P = sdiag(1.0 / self.dc_voltage) * P

        if self.storeProjections:
            self._Ps[mesh] = P

        return P


class Pole(BaseRx):
    """
    Pole receiver
    """

    # this should probably be updated to n_receivers...
    @property
    def nD(self):
        """Number of data in the receiver."""
        return self.locations.shape[0]

    nRx = deprecate_property(nD, "nRx", new_name="nD", removal_version="0.15.0")

    def getP(self, mesh, Gloc):
        if mesh in self._Ps:
            return self._Ps[mesh]

        P = mesh.getInterpolationMat(self.locations, Gloc)

        if self.data_type == "apparent_resistivity":
            P = sdiag(1.0 / self.geometric_factor) * P
        elif self.data_type == "apparent_chargeability":
            P = sdiag(1.0 / self.dc_voltage) * P

        if self.storeProjections:
            self._Ps[mesh] = P

        return P
