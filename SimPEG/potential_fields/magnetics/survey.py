import numpy as np
from ...survey import BaseSurvey
from ...utils.code_utils import validate_type
from .sources import UniformBackgroundField


class Survey(BaseSurvey):
    """Base Magnetics Survey

    Parameters
    ----------
    source_field : simpeg.potential_fields.magnetics.sources.UniformBackgroundField
        A source object that defines the Earth's inducing field
    """

    def __init__(self, source_field, **kwargs):
        self.source_field = validate_type(
            "source_field", source_field, UniformBackgroundField, cast=False
        )
        super().__init__(source_list=None, **kwargs)

    def eval(self, fields):  # noqa: A003
        """Compute the fields

        Parameters
        ----------
        fields : numpy.ndarray
            For the magnetics simulation, *fields* are the simulated response
            at the receiver locations.

        Returns
        -------
        numpy.ndarray
            Returns the input *fields*
        """
        return fields

    @property
    def nRx(self):
        """Total number of receivers.

        Returns
        -------
        int
            Total number of receivers
        """
        return sum(rx.locations.shape[0] for rx in self.source_field.receiver_list)

    @property
    def receiver_locations(self):
        """Receiver locations.

        Returns
        -------
        (n_loc, 3) numpy.ndarray
            Receiver locations
        """
        return np.concatenate([rx.locations for rx in self.source_field.receiver_list])

    @property
    def nD(self):
        """Total number of data.

        Returns
        -------
        int
            Total number of data (n_locations X n_components)
        """
        return sum(rx.nD for rx in self.source_field.receiver_list)

    @property
    def components(self):
        """Field components

        Returns
        -------
        list of str
            Components of the field being measured
        """
        comps = []
        for rx in self.source_field.receiver_list:
            comps += rx.components
        return comps

    def _location_component_iterator(self):
        for rx in self.source_field.receiver_list:
            for loc in rx.locations:
                yield loc, rx.components

    @property
    def vnD(self):
        """Vector number of data

        Returns
        -------
        list of int
            The number of data for each receivers.
        """
        return self.source_field.vnD


# make this look like it lives in the below module
Survey.__module__ = "simpeg.potential_fields.magnetics"
