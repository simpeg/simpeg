import numpy as np
import properties
from six import integer_types
import warnings

from .survey import BaseSurvey
from . import survey
from .utils import mkvc

__all__ = ["Data", "SyntheticData"]


class UncertaintyArray(properties.Array):

    class_info = "An array that can be set by a scalar value or numpy array"

    def validate(self, instance, value):
        if isinstance(value, integer_types):
            return float(value)
        elif isinstance(value, float):
            return value
        return super(properties.Array, self).validate(instance, value)


class Data(properties.HasProperties):
    """
    Data storage. This class keeps track of observed data, relative error
    of those data and the noise floor.

    .. code:: python

        data = Data(survey, dobs=dobs, relative_error=relative, noise_floor=floor)

    or

    .. code:: python

        data = Data(survey, dobs=dobs, standard_deviation=standard_deviation)
    """

    dobs = properties.Array(
        """
        Vector of the observed data. The data can be set using the survey
        parameters:

        .. code:: python

            data = Data(survey)
            for src in survey.source_list:
                for rx in src.receiver_list:
                    data[src, rx] = datum

        """,
        shape=("*",),
        required=True,
    )

    relative_error = UncertaintyArray(
        """
        Relative error of the data. This can be set using an array of the
        same size as the data (e.g. if you want to assign a different relative
        error to each datum) or as a scalar if you would like to assign a
        the same relative error to all data.

        The standard_deviation is constructed as follows::

            sqrt( (relative_error * np.abs(dobs))**2 + noise_floor**2 )

        For example, if you set

        .. code:: python

            data = Data(survey, dobs=dobs)
            data.relative_error = 0.05

        then the contribution to the standard_deviation is equal to

        .. code:: python

            data.relative_error * np.abs(data.dobs)

        """,
        shape=("*",),
    )

    noise_floor = UncertaintyArray(
        """
        Noise floor of the data. This can be set using an array of the
        same size as the data (e.g. if you want to assign a different noise
        floor to each datum) or as a scalar if you would like to assign a
        the same noise floor to all data.

        The standard_deviation is constructed as follows::

            sqrt( (relative_error * np.abs(dobs))**2 + noise_floor**2 )

        For example, if you set

        .. code:: python

            data = Data(survey, dobs=dobs)
            data.noise_floor = 1e-10

        then the contribution to the standard_deviation is equal to

        .. code:: python

            data.noise_floor

        """,
        shape=("*",),
    )

    survey = properties.Instance("a SimPEG survey object", BaseSurvey, required=True)

    _uid = properties.Uuid("unique ID for the data")

    #######################
    # Instantiate the class
    #######################
    def __init__(
        self,
        survey,
        dobs=None,
        relative_error=None,
        noise_floor=None,
        standard_deviation=None,
    ):
        super(Data, self).__init__()
        self.survey = survey

        # Observed data
        if dobs is None:
            dobs = np.nan * np.ones(survey.nD)  # initialize data as nans
        if isinstance(dobs, dict):
            # Check if survey has a components attribute
            if not hasattr(survey, "components"):
                raise ValueError(
                    """
                    Impossible to take 'dobs' as a dictionary since the passed
                    survey has not a 'components' attribute.
                    """
                )
            # Check if dobs has invalid or missing components
            _check_invalid_and_missing_components(dobs.keys(), survey.components.keys())
            # Check if all elements in dobs have the same size
            _check_data_sizes(dobs)
            # Build the dobs array (data values for each component should be
            # interleaved)
            dobs = _observed_data_dict_to_array(dobs, survey.components.keys())
        self.dobs = dobs

        if relative_error is not None:
            self.relative_error = relative_error

        if noise_floor is not None:
            self.noise_floor = noise_floor

        if standard_deviation is not None:
            if relative_error is not None or noise_floor is not None:
                warnings.warn(
                    "Setting the standard_deviation overwrites the "
                    "relative_error and noise_floor"
                )
            self.standard_deviation = standard_deviation

        if (
            standard_deviation is None
            and relative_error is None
            and noise_floor is None
        ):
            self.standard_deviation = 0.0

    #######################
    # Properties
    #######################
    @property
    def standard_deviation(self):
        """
        Data standard deviations. If a relative error and noise floor are
        provided, the standard_deviation is

        .. code:: python

            data.standard_deviation = np.sqrt(
                (data.relative_error*np.abs(data.dobs))**2 +
                data.noise_floor**2
            )

        otherwise, the standard_deviation can be set directly

        .. code:: python

            data.standard_deviation = 0.05 * np.absolute(self.dobs) + 1e-12

        Note that setting the standard_deviation directly will clear the `relative_error`
        and set the value to the `noise_floor` property.

        """
        if self.relative_error is None and self.noise_floor is None:
            raise Exception(
                "The relative_error and / or noise_floor must be set "
                "before asking for uncertainties. Alternatively, the "
                "standard_deviation can be set directly"
            )

        uncert = np.zeros(self.nD)
        if self.relative_error is not None:
            uncert += np.array(self.relative_error * np.absolute(self.dobs)) ** 2
        if self.noise_floor is not None:
            uncert += np.array(self.noise_floor) ** 2

        return np.sqrt(uncert)

    @standard_deviation.setter
    def standard_deviation(self, value):
        self.relative_error = np.zeros(self.nD)
        self.noise_floor = value

    @property
    def nD(self):
        return len(self.dobs)

    @property
    def shape(self):
        return self.dobs.shape

    ##########################
    # Observers and validators
    ##########################

    @properties.validator("dobs")
    def _dobs_validator(self, change):
        if self.survey.nD != len(change["value"]):
            raise ValueError(
                "{} must have the same length as the number of data. The "
                "provided input has len {}, while the survey expects "
                "survey.nD = {}".format(
                    change["name"], len(change["value"]), self.survey.nD
                )
            )

    @properties.validator(["relative_error", "noise_floor"])
    def _standard_deviation_validator(self, change):
        if isinstance(change["value"], float):
            change["value"] = change["value"] * np.ones(self.nD)
        self._dobs_validator(change)

    @property
    def index_dictionary(self):
        """
        Dictionary of data indices by sources and receivers. To set data using
        survey parameters:

        .. code:: python

            data = Data(survey)
            for src in survey.source_list:
                for rx in src.receiver_list:
                    index = data.index_dictionary[src][rx]
                    data.dobs[index] = datum

        """
        if getattr(self, "_index_dictionary", None) is None:
            if self.survey is None:
                raise Exception(
                    "To set or get values by source-receiver pairs, a survey must "
                    "first be set. `data.survey = survey`"
                )

            # create an empty dict
            self._index_dictionary = {}

            # create an empty dict associated with each source
            for src in self.survey.source_list:
                self._index_dictionary[src] = {}

            # loop over sources and find the associated data indices
            indBot, indTop = 0, 0
            for src in self.survey.source_list:
                for rx in src.receiver_list:
                    indTop += rx.nD
                    self._index_dictionary[src][rx] = np.arange(indBot, indTop)
                    indBot += rx.nD

        return self._index_dictionary

    ##########################
    # Methods
    ##########################

    def __setitem__(self, key, value):
        index = self.index_dictionary[key[0]][key[1]]
        self.dobs[index] = mkvc(value)

    def __getitem__(self, key):
        index = self.index_dictionary[key[0]][key[1]]
        return self.dobs[index]

    def tovec(self):
        return self.dobs

    def fromvec(self, v):
        v = mkvc(v)
        self.dobs = v


class SyntheticData(Data):
    """
    Data class for synthetic data. It keeps track of observed and clean data
    """

    dclean = properties.Array(
        """
        Vector of the clean synthetic data. The data can be set using the survey
        parameters:

        .. code:: python

            data = Data(survey)
            for src in survey.source_list:
                for rx in src.receiver_list:
                    index = data.index_dictionary(src, rx)
                    data.dclean[indices] = datum

        """,
        shape=("*",),
        required=True,
    )

    def __init__(
        self, survey, dobs=None, dclean=None, relative_error=None, noise_floor=None
    ):
        super(SyntheticData, self).__init__(
            survey=survey,
            dobs=dobs,
            relative_error=relative_error,
            noise_floor=noise_floor,
        )

        if dclean is None:
            dclean = np.nan * np.ones(self.survey.nD)
        self.dclean = dclean

    @properties.validator("dclean")
    def _dclean_validator(self, change):
        self._dobs_validator(change)


# inject a new data class into the survey module
class _Data(Data):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "The survey.Data class has been moved. To import the data class, "
            "please use SimPEG.data.Data."
        )


def _check_invalid_and_missing_components(data_components, survey_components):
    """
    Check for invalid and missing components in dobs dictionary

    Parameters
    ----------
    data_components : dict_keys
        Dictionary keys containing the components in the data.
    survey_components : dict_keys
        Dictionary keys containing the components present in the survey.
    """
    invalid_components = [c for c in data_components if c not in survey_components]
    if invalid_components:
        invalid_string = "', '".join(invalid_components)
        survey_string = "', '".join(survey_components)
        raise ValueError(
            f"Found invalid components '{invalid_string}' "
            + "in the dobs dictionary. "
            + "For the current survey, dobs needs to have the following "
            + f"components: '{survey_string}'."
        )
    missing_components = [c for c in survey_components if c not in data_components]
    if missing_components:
        missing_string = "', '".join(missing_components)
        raise ValueError(f"Missing '{missing_string}' components in dobs dictionary.")


def _check_data_sizes(observed_data):
    """
    Check that all elements inside observed data dictionary have the same shape
    """
    # Check if every array in observed_data has the same size
    sizes = np.array([d.size for d in observed_data.values()])
    if not (sizes[0] == sizes).all():
        raise ValueError(
            "All elements in the data dictionary should have the same size"
        )


def _observed_data_dict_to_array(observed_data, survey_components):
    """
    Convert a dictionary with observed data (or uncertainties) into a 1d array

    The generated array interleaves the values of each component: we list
    all the values of the observed components (or their uncertainties) on
    a single receiver before moving to the next one (e.g. we loop faster on
    components than on receivers).

    Parameters
    ----------
    observed_data : dict
        Dictionary containing the data values for each one of the observed
        components.
    survey_components : dict_keys
        Dictionary keys containing the components present in the survey.

    Parameters
    ----------
    array : 1d-array
        1D array with all the components of the observed data (or their
        uncertainties).
    """
    # Determine the size of the full array
    n_components = len(survey_components)
    size_of_single_component = list(observed_data.values())[0].size
    size = n_components * size_of_single_component
    # Allocate the full array
    dobs = np.empty(size, dtype=np.float64)
    # Fill the dobs arrays interleaving the components.
    # It's important to iterate in the same order in which the survey
    # components are defined!
    for i, component in enumerate(survey_components):
        dobs[i::n_components] = observed_data[component]
    return dobs


survey.Data = _Data
survey.Data.__name__ = "Data"
survey.Data.__qualname__ = "Data"
survey.Data.__module__ = "SimPEG.survey"
