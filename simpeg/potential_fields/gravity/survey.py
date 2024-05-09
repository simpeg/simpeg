from ...survey import BaseSurvey
from ...utils.code_utils import validate_type
from .sources import SourceField


class Survey(BaseSurvey):
    """Base Gravity Survey

    Parameters
    ----------
    source_field : simpeg.potential_fields.gravity.sources.SourceField
        A source object that defines receivers locations for gravity.
    """

    def __init__(self, source_field, **kwargs):
        self.source_field = validate_type(
            "source_field", source_field, SourceField, cast=False
        )
        super().__init__(source_list=None, **kwargs)

    def eval(self, fields):  # noqa: A003
        """Evaluate the field

        Parameters
        ----------
        fields : numpy.ndarray
            The potential fields field object

        Returns
        -------
        numpy.ndarray
            the fields at the receiver locations.
        """
        return fields

    @property
    def nRx(self):
        """Returns the number of receivers in the survey

        Returns
        -------
        int
            Number of receivers in the survey
        """
        return sum(
            receiver.locations.shape[0] for receiver in self.source_field.receiver_list
        )

    @property
    def receiver_locations(self):
        """Receiver locations.

        Returns the receiver locations for the survey as a (n_loc, 3) ``numpy.ndarray``;
        even if multiple components (e.g. 'gx', 'gxy', 'gzz') are measured at each location.

        Returns
        -------
        (n_loc, 3) numpy.ndarray
            Receiver locations.
        """
        return self.source_field.receiver_list[0].locations

    @property
    def nD(self):
        """Number of data for the survey.

        Returns the total number of data; i.e. the number of locations time the
        number of components.

        Returns
        -------
        int
            Number of data for the survey
        """
        return sum(receiver.nD for receiver in self.source_field.receiver_list)

    @property
    def components(self):
        """Number of components measured at each receiver.

        Returns
        -------
        int
            Number of components measured at each receiver.
        """
        return self.source_field.receiver_list[0].components

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

    @property
    def Qfx(self):
        """Projection matrix from x-faces to receiver locations.

        Returns
        -------
        scipy.sparse.csv_matrix
            The projection matrix from x-faces to receiver locations
        """
        if getattr(self, "_Qfx", None) is None:
            self._Qfx = self.prob.mesh.get_interpolation_matrix(
                self.receiver_locations, "Fx"
            )
        return self._Qfx

    @property
    def Qfy(self):
        """Projection matrix from y-faces to receiver locations.

        Returns
        -------
        scipy.sparse.csv_matrix
            The projection matrix from y-faces to receiver locations
        """
        if getattr(self, "_Qfy", None) is None:
            self._Qfy = self.prob.mesh.get_interpolation_matrix(
                self.receiver_locations, "Fy"
            )
        return self._Qfy

    @property
    def Qfz(self):
        """Projection matrix from z-faces to receiver locations.

        Returns
        -------
        scipy.sparse.csv_matrix
            The projection matrix from z-faces to receiver locations
        """
        if getattr(self, "_Qfz", None) is None:
            self._Qfz = self.prob.mesh.get_interpolation_matrix(
                self.receiver_locations, "Fz"
            )
        return self._Qfz

    def projectFields(self, u):
        r"""Project the fields from the mesh to the receiver locations.


        Parameters
        ----------
        u : dict of numpy.ndarray
            Field object created from a `gravity.Simulation3DDifferential`.

        Returns
        -------
        dict
            gx, gy, and gz Projected fields

        """
        # TODO: There can be some different tyes of data like |B| or B

        gfx = self.Qfx * u["G"]
        gfy = self.Qfy * u["G"]
        gfz = self.Qfz * u["G"]

        fields = {"gx": gfx, "gy": gfy, "gz": gfz}
        return fields
