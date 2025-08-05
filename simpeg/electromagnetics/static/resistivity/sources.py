import numpy as np
from .... import survey
from ....utils import Zero, validate_ndarray_with_shape


class BaseSrc(survey.BaseSrc):
    """Base DC/IP source

    Parameters
    ----------
    receiver_list : list of simpeg.electromagnetics.static.resistivity.receivers.BaseRx
        A list of DC/IP receivers
    location : (n_source, dim) numpy.ndarray
        Source locations
    current : float or numpy.ndarray, default: 1.0
        Current amplitude [A]
    """

    _q = None

    def __init__(self, receiver_list, location, current=1.0, **kwargs):
        super().__init__(receiver_list=receiver_list, **kwargs)
        self.location = location
        self.current = current

    @property
    def location(self):
        """locations of the source electrodes

        Returns
        -------
        (n, dim) numpy.ndarray
            Locations of the source electrodes
        """
        return self._location

    @location.setter
    def location(self, other):
        self._location = validate_ndarray_with_shape(
            "location", other, shape=("*", "*"), dtype=float
        )

    @property
    def current(self):
        """Amplitudes of the source currents

        Returns
        -------
        (n_source) numpy.ndarray
            Amplitudes of the source currents
        """
        return self._current

    @current.setter
    def current(self, other):
        other = validate_ndarray_with_shape("current", other, shape=("*",), dtype=float)
        if len(other) > 1 and len(other) != self.location.shape[0]:
            raise ValueError(
                "Current must be constant or equal to the number of specified source locations."
                f" saw {len(other)} current sources and {self.location.shape[0]} locations."
            )
        self._current = other

    def eval(self, sim):  # noqa: A003
        """Discretize sources to mesh

        Parameters
        ----------
        sim : simpeg.base.BaseElectricalPDESimulation
            The static electromagnetic simulation

        Returns
        -------
        numpy.ndarray
            The right-hand sides corresponding to the sources
        """
        if sim._formulation == "HJ":
            inds = sim.mesh.closest_points_index(self.location, grid_loc="CC")
            q = np.zeros(sim.mesh.nC)
            q[inds] = self.current
        elif sim._formulation == "EB":
            loc = self.location
            cur = self.current
            interpolation_matrix = sim.mesh.get_interpolation_matrix(
                loc, location_type="N"
            ).toarray()
            q = np.sum(cur[:, np.newaxis] * interpolation_matrix, axis=0)
        return q

    def evalDeriv(self, sim):
        """Returns the derivative of the source term with respect to the model.

        This is zero.

        Parameters
        ----------
        sim : simpeg.base.BaseElectricalPDESimulation
            The static electromagnetic simulation

        Returns
        -------
        discretize.utils.Zero
        """
        return Zero()


class Multipole(BaseSrc):
    """
    Generic Multipole Source
    """

    @property
    def location_a(self):
        """Locations of the A electrodes

        Returns
        -------
        (n, dim) numpy.ndarray
            Locations of the A electrodes
        """
        return self.location

    @property
    def location_b(self):
        """Location of the B electrodes

        Returns
        -------
        (n, dim) numpy.ndarray
            Locations of the B electrodes
        """
        return np.full_like(self.location, np.nan)


class Dipole(BaseSrc):
    """Dipole source

    Parameters
    ----------
    receiver_list : list of simpeg.electromagnetics.static.resistivity.receivers.BaseRx
        A list of DC/IP receivers
    location_a : (dim) array_like
        A electrode locations; remember to set ``location_b`` keyword argument
        to define B electrode location.
    location_b : (dim) array_like
        B electrode locations; remember to set ``location_a`` keyword argument
        to define A electrode location.
    location : tuple of array_like, optional
        A and B electrode locations. If ``location_a`` and ``location_b`` are
        provided, don't pass values to this argument. Otherwise, provide
        a tuple of the form ``(location_a, location_b)``.
    current : float, optional
        Current amplitude in :math:`A` that goes through each electrode.
    """

    def __init__(
        self,
        receiver_list,
        location_a=None,
        location_b=None,
        location=None,
        current=1.0,
    ):
        if location is None and location_a is None and location_b is None:
            raise TypeError(
                "Found 'location', 'location_a' and 'location_b' as None. "
                "Please specify 'location', or 'location_a' and 'location_b' "
                "when defining a dipole source."
            )
        if location is not None and (location_a is not None or location_b is not None):
            raise TypeError(
                "Found 'location_a' and/or 'location_b' as not None values. "
                "When passing a not None value for 'location', 'location_a' and "
                "'location_b' should be set to None."
            )
        if location is None:
            if location_a is None:
                raise TypeError(
                    "Invalid 'location_a' set to None. When 'location' is None, "
                    "both 'location_a' and 'location_b' should be set to "
                    "a value different than None."
                )
            if location_b is None:
                raise TypeError(
                    "Invalid 'location_b' set to None. When 'location' is None, "
                    "both 'location_a' and 'location_b' should be set to "
                    "a value different than None."
                )
            location = [location_a, location_b]

        if len(location) != 2:
            raise ValueError(
                "location must be a list or tuple of length 2: "
                "[location_a, location_b]. The input location has "
                f"length {len(location)}"
            )

        super().__init__(
            receiver_list=receiver_list,
            location=location,
            current=[current, -current],
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(" f"a: {self.location_a}; b: {self.location_b})"
        )

    @property
    def location_a(self):
        """Locations of the A-electrodes

        Returns
        -------
        (n_source, dim) numpy.ndarray
            Locations of the A-electrodes
        """
        return self.location[0]

    @property
    def location_b(self):
        """Locations of the B-electrodes

        Returns
        -------
        (n_source, dim) numpy.ndarray
            Locations of the B-electrodes
        """
        return self.location[1]


class Pole(BaseSrc):
    """Pole source

    Parameters
    ----------
    receiver_list : list of simpeg.electromagnetics.static.resistivity.receivers.BaseRx
        A list of DC/IP receivers
    location : (n_source, dim) numpy.ndarray
        Electrode locations
    """

    def __init__(self, receiver_list, location=None, **kwargs):
        super().__init__(receiver_list=receiver_list, location=location, **kwargs)
        if len(self.location) != 1:
            raise ValueError(
                f"Pole sources only have a single location, not {len(self.location)}"
            )

    @property
    def location_a(self):
        """Locations of the A-electrodes

        Returns
        -------
        (n_source, dim) numpy.ndarray
            Locations of the A-electrodes
        """
        return self.location[0]

    @property
    def location_b(self):
        """Locations of the B-electrodes

        Returns
        -------
        (n_source, dim) numpy.ndarray of ``numpy.nan``
            Locations of the B-electrodes
        """
        return np.full_like(self.location[0], np.nan)
