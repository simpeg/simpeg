import numpy as np
import properties

from .... import survey
from ....utils import Zero


class BaseSrc(survey.BaseSrc):
    """
    Base DC source
    """

    _q = None

    def __init__(self, receiver_list, location, current=1.0, **kwargs):
        super().__init__(receiver_list=receiver_list, **kwargs)
        self.location = location
        self.current = current

    @property
    def location(self):
        """location of the source electrodes"""
        return self._location

    @location.setter
    def location(self, other):
        other = np.asarray(other, dtype=float)
        other = np.atleast_2d(other)
        self._location = other

    @property
    def current(self):
        """amplitudes of the source currents"""
        return self._current

    @current.setter
    def current(self, other):
        other = np.atleast_1d(np.asarray(other, dtype=float))
        if other.ndim > 1:
            raise ValueError("Too many dimensions for current array")
        if len(other) > 1 and len(other) != self.location.shape[0]:
            raise ValueError(
                "Current must be constant or equal to the number of specified source locations."
                f" saw {len(other)} current sources and {self.location.shape[0]} locations."
            )
        self._current = other

    def eval(self, sim):
        if self._q is not None:
            return self._q
        else:
            if sim._formulation == "HJ":
                inds = sim.mesh.closest_points_index(self.location, grid_loc="CC")
                self._q = np.zeros(sim.mesh.nC)
                self._q[inds] = self.current
            elif sim._formulation == "EB":
                loc = self.location
                cur = self.current
                interpolation_matrix = sim.mesh.get_interpolation_matrix(
                    loc, locType="N"
                ).toarray()
                q = np.sum(cur[:, np.newaxis] * interpolation_matrix, axis=0)
                self._q = q
            return self._q

    def evalDeriv(self, sim):
        return Zero()


class Multipole(BaseSrc):
    """
    Generic Multipole Source
    """

    @property
    def location_a(self):
        """Locations of the A electrode"""
        return self.location

    @property
    def location_b(self):
        """Location of the B electrode"""
        return np.full_like(self.location, np.nan)


class Dipole(BaseSrc):
    """
    Dipole source
    """

    def __init__(
        self, receiver_list, location_a=None, location_b=None, location=None, **kwargs,
    ):
        if "current" in kwargs.keys():
            value = kwargs.pop("current")
            current = [value, -value]
        else:
            current = [1.0, -1.0]

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

        # instantiate
        super().__init__(
            receiver_list=receiver_list, location=location, current=current, **kwargs
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(" f"a: {self.location_a}; b: {self.location_b})"
        )

    @property
    def location_a(self):
        """Location of the A-electrode"""
        return self.location[0]

    @property
    def location_b(self):
        """Location of the B-electrode"""
        return self.location[1]


class Pole(BaseSrc):
    def __init__(self, receiver_list, location=None, **kwargs):
        super().__init__(receiver_list=receiver_list, location=location, **kwargs)
        if len(self.location) != 1:
            raise ValueError(
                f"Pole sources only have a single location, not {len(self.location)}"
            )

    @property
    def location_a(self):
        """Location of the A electrode"""
        return self.location[0]

    @property
    def location_b(self):
        """Location of the B electrode"""
        return np.full_like(self.location[0], np.nan)
