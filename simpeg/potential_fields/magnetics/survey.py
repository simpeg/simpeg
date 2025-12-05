import numpy as np
from ...survey import BaseSurvey
from ...utils.code_utils import validate_list_of_types
from .sources import UniformBackgroundField


class Survey(BaseSurvey):
    """Base Magnetics Survey

    Parameters
    ----------
    source_field : simpeg.potential_fields.magnetics.sources.UniformBackgroundField
        A source object that defines the Earth's inducing field
    """

    def __init__(self, source_field, **kwargs):
        if "source_list" in kwargs:
            msg = (
                "source_list is not a valid argument to gravity.Survey. "
                "Use source_field instead."
            )
            raise TypeError(msg)
        super().__init__(source_list=source_field, **kwargs)

    @BaseSurvey.source_list.setter
    def source_list(self, new_list):
        # mag simulations only support 1 source... for now...
        self._source_list = validate_list_of_types(
            "source_list",
            new_list,
            UniformBackgroundField,
            ensure_unique=True,
            min_n=1,
            max_n=1,
        )

    @property
    def source_field(self):
        """A source defining the Earth's inducing field and containing the magnetic receivers.

        Returns
        -------
        simpeg.potential_fields.magnetics.sources.UniformBackgroundField
        """
        return self.source_list[0]

    @source_field.setter
    def source_field(self, new_src):
        self.source_list = new_src

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
