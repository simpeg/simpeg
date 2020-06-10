import numpy as np
import properties
from six import integer_types
import warnings

from .survey import BaseSurvey
from . import survey
from .utils import mkvc
from .utils.code_utils import deprecate_property

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

            relative_error * np.abs(dobs) + noise_floor

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

            relative_error * np.abs(dobs) + noise_floor

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

            data.standard_deviation = (
                data.relative_error*np.abs(data.dobs) +
                data.noise_floor
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
            uncert = uncert + self.relative_error * np.absolute(self.dobs)
        if self.noise_floor is not None:
            uncert = uncert + self.noise_floor

        return uncert

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

    ##########################
    # Deprecated
    ##########################
    std = deprecate_property(
        relative_error, "std", new_name="relative_error", removal_version="0.15.0"
    )
    eps = deprecate_property(
        noise_floor, "eps", new_name="noise_floor", removal_version="0.15.0"
    )


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
        warnings.warn(
            "The survey.Data class has been moved. To import the data class, "
            "please use SimPEG.data.Data. This class will be removed in SimPEG 0.15.0",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


survey.Data = _Data
survey.Data.__name__ = "Data"
survey.Data.__qualname__ = "Data"
survey.Data.__module__ = "SimPEG.survey"
