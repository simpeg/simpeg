import numpy as np

# import properties

from .... import survey
from ....utils import Zero, closestPoints, mkvc, validate_list_of_types, validate_float
from .receivers import BaseRx


class BaseSrc(survey.BaseSrc):
    """Base spectral IP source

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.static.resistivity.receivers.BaseRx
        A list of DC/IP receivers
    location : (dim) numpy.ndarray
        Source location
    current : float, default=1.0
        Current amplitude [A]
    """

    # current = properties.Float("Source current", default=1.0)

    def __init__(self, receiver_list, location, current=1.0, **kwargs):
        super(BaseSrc, self).__init__(
            receiver_list=receiver_list, location=location, **kwargs
        )
        self.current = current

    @property
    def receiver_list(self):
        """List of receivers associated with the source

        Returns
        -------
        list of SimPEG.electromagnetics.static.spectral_induced_polarization.receivers.BaseRx
            List of receivers associated with the source
        """
        return self._receiver_list

    @receiver_list.setter
    def receiver_list(self, new_list):
        self._receiver_list = validate_list_of_types(
            "source_list", new_list, BaseRx, ensure_unique=True
        )

    @property
    def current(self):
        """Source current

        Returns
        -------
        float
            Source current
        """
        return self._current

    @current.setter
    def current(self, I):
        I = validate_float("current", I)
        if I == 0.0:
            raise ValueError("current must be non-zero.")
        self._current = I

    def eval(self, simulation):
        """Not implemented for BaseSrc"""
        raise NotImplementedError

    def evalDeriv(self, simulation):
        """Returns zero"""
        return Zero()

    @property
    def nD(self):
        """Number of data

        Returns
        -------
        int
            Number of data
        """
        return self.vnD.sum()

    @property
    def vnD(self):
        """Vector number of data

        Returns
        -------
        list of int
            Number of data per source
        """
        return np.array([rx.nD * len(rx.times) for rx in self.receiver_list])


class Dipole(BaseSrc):
    """Spectral IP dipole source

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.static.spectral_induced_polarization.receivers.BaseRx
        A list of spectral IP receivers
    location_a : (dim) numpy.ndarray
        A electrode location; remember to set 'location_b' keyword argument to define N electrode locations.
    location_b : (dim) numpy.ndarray
        B electrode location; remember to set 'location_a' keyword argument to define M electrode locations.
    location : list or tuple of length 2 of numpy.ndarray
        A and B electrode locations. In this case, do not set the 'location_a' and 'location_b'
        keyword arguments. And we supply a list or tuple of the form [location_a, location_b].
    current : float, default=1.0
        Current amplitude [A]
    """

    # location = properties.List(
    #     "location of the source electrodes",
    #     survey.SourceLocationArray("location of electrode"),
    # )

    def __init__(
        self,
        receiver_list=None,
        location_a=None,
        location_b=None,
        location=None,
        **kwargs,
    ):
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

        if location is None:
            raise AttributeError(
                "Source cannot be instantiated without assigning 'location'."
                "Please provide either location=(location_a, location_b) "
                "or both location_a=location_a, location_b=location_b"
            )

        super(Dipole, self).__init__(receiver_list, location=location, **kwargs)

    # @property
    # def location_a(self):
    #     """Location of the A-electrode"""
    #     return self.location[0]

    # @property
    # def location_b(self):
    #     """Location of the B-electrode"""
    #     return self.location[1]

    @property
    def location(self):
        """A and N electrode locations

        Returns
        -------
        list of np.ndarray of the form [dim, dim]
            A and B electrode locations.
        """
        return self._locations

    @location.setter
    def location(self, locs):
        if len(locs) != 2:
            raise ValueError(
                "locations must be a list or tuple of length 2: "
                "[location_a, location_b. The input locations has "
                f"length {len(locs)}"
            )

        locs = [np.atleast_1d(locs[0]), np.atleast_1d(locs[1])]

        # check the size of locations_m, locations_n
        if locs[0].shape != locs[1].shape:
            raise ValueError(
                f"location_a (shape: {locs[0].shape}) and "
                f"location_b (shape: {locs[1].shape}) need to be "
                f"the same size"
            )

        self._locations = locs

    @property
    def location_a(self):
        """Location of the A-electrode

        Returns
        -------
        (dim) numpy.ndarray
            Location of the A-electrode
        """
        return self.location[0]

    @property
    def location_b(self):
        """Location of the B-electrode

        Returns
        -------
        (dim) numpy.ndarray
            Location of the B-electrode
        """
        return self.location[1]

    def eval(self, simulation):
        """Project source to mesh.

        Parameters
        ----------
        sim : SimPEG.electromagnetics.static.spectral_induced_polarization.simulation.BaseDCSimulation
            A spectral IP simulation

        Returns
        -------
        numpy.ndarray
            Discretize source term on the mesh
        """
        if simulation._formulation == "HJ":
            inds = closestPoints(simulation.mesh, self.location, gridLoc="CC")
            q = np.zeros(simulation.mesh.nC)
            q[inds] = self.current * np.r_[1.0, -1.0]
        elif simulation._formulation == "EB":
            qa = simulation.mesh.getInterpolationMat(
                self.location[0], locType="N"
            ).todense()
            qb = -simulation.mesh.getInterpolationMat(
                self.location[1], locType="N"
            ).todense()
            q = self.current * mkvc(qa + qb)
        return q


class Pole(BaseSrc):
    """Spectral IP pole source

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.static.spectral_induced_polarization.receivers.BaseRx
        list of spectral IP receivers
    location : (dim) array_like
        Electrode location
    current : float, default=1.0
        Current amplitude [A]
    """

    def __init__(self, receiver_list=None, location=None, **kwargs):
        super(Pole, self).__init__(receiver_list, location=location, **kwargs)

    def eval(self, simulation):
        """Project source to mesh.

        Parameters
        ----------
        sim : SimPEG.electromagnetics.static.resistivity.simulation.BaseDCSimulation
            A DC/IP simulation

        Returns
        -------
        numpy.ndarray
            Discretize source term on the mesh
        """
        if simulation._formulation == "HJ":
            inds = closestPoints(simulation.mesh, self.location)
            q = np.zeros(simulation.mesh.nC)
            q[inds] = self.current * np.r_[1.0]
        elif simulation._formulation == "EB":
            q = simulation.mesh.getInterpolationMat(
                self.location, locType="N"
            ).todense()
            q = self.current * mkvc(q)
        return q
