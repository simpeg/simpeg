import numpy as np
import properties

from .... import survey
from ....utils import Zero, closestPoints


class BaseSrc(survey.BaseSrc):
    """
    Base DC source
    """

    current = properties.List(doc="amplitudes of the source currents", default=[1.0])
    location = properties.List(
        "location of the source electrodes",
        survey.SourceLocationArray("location of electrode"),
    )

    _q = None

    def __init__(self, receiver_list, location, current=None, **kwargs):
        super(BaseSrc, self).__init__(receiver_list, **kwargs)
        if type(location) is np.ndarray:
            if location.ndim == 2:
                location = list(location)
            else:
                location = [location]
        if current is None:
            current = self.current
        if type(current) == float or type(current) == int:
            current = [float(current)]
        elif type(current) == np.ndarray:
            if current.ndim == 2:
                current = list(current)
            else:
                location = [location]
        if len(current) != 1 and len(current) != len(location):
            raise ValueError(
                "Current must be constant or equal to the number of specified source locations."
            )
        if type(current) != list:
            current = np.repeat(current, len(location)).tolist()

        self.current = current
        self.location = location

    def eval(self, sim):
        if self._q is not None:
            return self._q
        else:
            if sim._formulation == "HJ":
                inds = sim.mesh.closest_points_index(self.location, grid_loc="CC")
                self._q = np.zeros(sim.mesh.nC)
                self._q[inds] = self.current
            elif sim._formulation == "EB":
                loc = np.row_stack(self.location)
                cur = np.asarray(self.current)
                interpolation_matrix = sim.mesh.get_interpolation_matrix(loc, locType='N').toarray()
                q = np.sum(cur[:, np.newaxis] * interpolation_matrix, axis=0)
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

    @property
    def location_a(self):
        """Locations of the A electrode"""
        return self.location

    @property
    def location_b(self):
        """Location of the B electrode"""
        return list(np.tile((len(self.location), 1),
                            np.full_like(self.location[0], np.nan)
                            )
                    )


class Dipole(BaseSrc):
    """
    Dipole source
    """

    location = properties.List(
        "location of the source electrodes",
        survey.SourceLocationArray("location of electrode"),
    )

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
            raise TypeError(
                "The locationA property has been removed. Please set the "
                "location_a property instead.",
            )

        if "locationB" in kwargs.keys():
            location_b = kwargs.pop("locationB")
            raise TypeError(
                "The locationB property has been removed. Please set the "
                "location_b property instead.",
            )
        if "current" in kwargs.keys():
            value = kwargs.pop('current')
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

        if location[0].shape != location[1].shape:
            raise ValueError(
                f"m_location (shape: {location[0].shape}) and "
                f"n_location (shape: {location[1].shape}) need to be "
                f"the same size"
            )

        # instantiate
        super(Dipole, self).__init__(receiver_list, location, current, **kwargs)

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
    def __init__(self, receiver_list=[], location=None, **kwargs):
        super(Pole, self).__init__(receiver_list, location, **kwargs)

    @property
    def location_a(self):
        """Locations of the A electrode"""
        return self.location[0]

    @property
    def location_b(self):
        """Location of the B electrode"""
        return np.nan * np.ones_like(self.location[0])
