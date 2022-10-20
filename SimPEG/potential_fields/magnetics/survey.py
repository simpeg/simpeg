import numpy as np
from ...survey import BaseSurvey
from ...utils.code_utils import validate_type
from .sources import SourceField


class Survey(BaseSurvey):
    """Base Magnetics Survey

    Parameters
    ----------
    source_field : SimPEG.potential_fields.magnetics.sources.SourceField
        A source object that defines the Earth's inducing field
    """

    # source_field = properties.Instance(
    #     "The inducing field source for the survey",
    #     properties.Instance("A SimPEG source", SourceField),
    #     default=SourceField
    # )

    def __init__(self, source_field, **kwargs):
        self.source_field = validate_type(
            "source_field", source_field, SourceField, cast=False
        )
        super().__init__(source_list=None, **kwargs)

    def eval(self, fields):
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

        return self.source_field.receiver_list[0].locations.shape[0]

    @property
    def receiver_locations(self):
        """Receiver locations.

        Returns
        -------
        (n_loc, 3) numpy.ndarray
            Receiver locations
        """
        return self.source_field.receiver_list[0].locations

    @property
    def nD(self):
        """Total number of data.

        Returns
        -------
        int
            Total number of data (n_locations X n_components)
        """
        return len(self.receiver_locations) * len(self.components)

    @property
    def components(self):
        """Field components

        Returns
        -------
        list of str
            Components of the field being measured
        """
        return self.source_field.receiver_list[0].components

    @property
    def vnD(self):
        """Vector number of data

        Returns
        -------
        list of int
            The number of data for each receivers.
        """

        if getattr(self, "_vnD", None) is None:
            self._vnD = []
            for receiver in self.source_field.receiver_list:

                for component in list(receiver.components.keys()):

                    # If non-empty than logcial for empty entries
                    self._vnD.append(int(receiver.components[component].sum()))

            print(self._vnD)
            self._vnD = np.asarray(self._vnD)
        return self._vnD


# make this look like it lives in the below module
Survey.__module__ = "SimPEG.potential_fields.magnetics"
