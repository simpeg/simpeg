import numpy as np
import properties

from .... import survey
from ....utils import Zero, closestPoints
from ....utils.code_utils import deprecate_property

import warnings


class BaseSrc(survey.BaseSrc):
    """
    Generic Base DC source
    """

    current = properties.Array(doc="amplitudes of the source currents", dtype=float, default=np.array([1.0]))
    location = properties.List(
        "location of the source electrodes",
        survey.SourceLocationArray("location of electrode"),
    )
    loc = deprecate_property(
        location, "loc", new_name="location", removal_version="0.15.0"
    )

    _q = None

    def __init__(self, receiver_list, location, current=None, **kwargs):
        super(BaseSrc, self).__init__(receiver_list, **kwargs)
        if current is None:
            current = self.current[0]
        if len(current) != 1 and len(current) != len(location):
            raise ValueError(
                "Current must be constant or equal to the number of specified source locations."
            )
        if len(current) == 1:
            current = np.repeat(current, len(location))

        self.current = current
        self.location = location

    def eval(self, sim):
        if self._q is not None:
            return self._q
        else:
            if sim._formulation == "HJ":
                inds = closestPoints(sim.mesh, self.location, gridLoc="CC")
                self._q = np.zeros(sim.mesh.nC)
                self._q[inds] = self.current
            elif sim._formulation == "EB":
                q = 0.0
                for loc, cur in zip(self.location, self.current):
                    interpolation_matrix = sim.mesh.get_interpolation_matrix(loc, locType="N")
                    q += cur * interpolation_matrix.toarray()
                self._q = q
            return self._q

    def evalDeriv(self, sim):
        return Zero()


class Multipole(BaseSrc):
    """
    Generic Multipole Source
    """

    def __init__(self, receiver_list=[], location=None, **kwargs):
        super(Multipole, self).__init__(receiver_list, location, **kwargs)


class Dipole(BaseSrc):
    """
    Dipole source
    """

    def __init__(
            self,
            receiver_list=[],
            location_a=None,
            location_b=None,
            location=None,
            **kwargs,
    ):
        # Check for old keywords
        if "locationA" in kwargs.keys():
            location_a = kwargs.pop("locationA")
            warnings.warn(
                "The locationA property has been deprecated. Please set the "
                "location_a property instead. This will be removed in version"
                " 0.15.0 of SimPEG",
                DeprecationWarning,
            )

        if "locationB" in kwargs.keys():
            location_b = kwargs.pop("locationB")
            warnings.warn(
                "The locationB property has been deprecated. Please set the "
                "location_b property instead. This will be removed in version"
                " 0.15.0 of SimPEG",
                DeprecationWarning,
            )
        current = np.r_[1.0, -1.0]
        if "current" in kwargs.keys():
            current *= kwargs.pop('current')

        # if location_a set, then use location_a, location_b
        if location_a is not None:
            if location_b is None:
                raise ValueError(
                    "For a dipole source both location_a and location_b " "must be set"
                )

            if location is not None:
                raise ValueError(
                    "Cannot set both location and location_a, location_b. "
                    "Please provide either location=(location_a, location_b) "
                    "or both location_a=location_a, location_b=location_b"
                )

            location = [location_a, location_b]

        elif location is not None:
            if len(location) != 2:
                raise ValueError(
                    "location must be a list or tuple of length 2: "
                    "[location_a, location_b]. The input location has "
                    f"length {len(location)}"
                )

        if location[0].shape != location[1].shape:
            raise ValueError(
                f"m_location (shape: {location[0].shape}) and "
                f"n_location (shape: {location[1].shape}) need to be "
                f"the same size"
            )

        # instantiate
        super(Dipole, self).__init__(receiver_list, location, current, **kwargs)

    @property
    def location_a(self):
        """Location of the A-electrode"""
        return self.location[0]

    @property
    def location_b(self):
        """Location of the B-electrode"""
        return self.location[1]


class Pole(BaseSrc):
    """
    Pole source
    """

    def __init__(self, receiver_list=[], location=None, **kwargs):
        super(Pole, self).__init__(receiver_list, location, **kwargs)
