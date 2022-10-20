import numpy as np
import properties
from six import integer_types
import warnings

from .survey import BaseSurvey
from .utils import mkvc

__all__ = ["Data", "SyntheticData"]


class UncertaintyArray(properties.Array):
    """Class for data uncertainties.

    ``UncertaintyArray`` is a class that ensures data uncertainties may
    only be defined by as a :class:`scalar` or :class:`numpy.ndarray`.
    If the uncertainties are defined using a single :class:`scalar` value,
    that value defines the uncertainties for all observed data. If the
    uncertainties are defined by an :class:`numpy.ndarray`, a
    specific uncertainty is being applied to each datum in the observed
    data.
    """

    class_info = "An array that can be set by a scalar value or numpy array"

    def validate(self, instance, value):
        """Validate uncertainties array.

        Parameters
        ----------
        instance : class
            Class for numerical values; e.g. ``float``, ``double``, ...
        value : object
            An input quantity you wish to evaluate

        Returns
        -------
        float or (n) numpy.array
            If *value* is a single numerical value, the return is a float.
            If *value* is a numpy.array, the return is a numpy array.
        """
        if isinstance(value, integer_types):
            return float(value)
        elif isinstance(value, float):
            return value
        return super(properties.Array, self).validate(instance, value)


class Data(properties.HasProperties):
    r"""Class for defining data in SimPEG.

    The ``Data`` class is used to create an object which connects the survey geometry,
    observed data and data uncertainties.

    Parameters
    ----------
    survey : SimPEG.survey.BaseSurvey
        A SimPEG survey object. For each geophysical method, the survey object defines
        the survey geometry; i.e. sources, receivers, data type.
    dobs : (n) numpy.ndarray
        Observed data.
    relative_error : UncertaintyArray
        Assign relative uncertainties to the data using relative error; sometimes
        referred to as percent uncertainties. For each datum, we assume the
        standard deviation of Gaussian noise is the relative error times the
        absolute value of the datum; i.e. :math:`C_{err} \times |d|`.
    noise_floor : UncertaintyArray
        Assign floor/absolute uncertainties to the data. For each datum, we assume
        standard deviation of Gaussian noise is equal to *noise_floor*.
    standard_deviation : UncertaintyArray
        Directly define the uncertainties on the data by assuming we know the standard
        deviations of the Gaussian noise. This is essentially the same as *noise_floor*.
        If set however, this will override *relative_error* and *noise_floor*.

    Notes
    -----
    If *noise_floor* (:math:`\varepsilon_{floor}`) and *relative_error*
    (:math:`C_{err}`) are used to define the uncertainties on the data, then for
    each datum (:math:`d`), the total uncertainty is given by:

    .. math::
        \varepsilon = \sqrt{\varepsilon_{floor}^2 + \big ( C_{err} |d| \big )^2}

    By using *standard_deviation* to assign the uncertainties, we are effectively
    providing :math:`\varepsilon` directly.

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
        r"""Return data uncertainties; i.e. the estimates of the standard deviations of the noise.

        If the *standard_deviation* property has already been set, it is returned
        directly. However if **standard_deviations** is ``None``, then uncertainties
        are computed and returned according to the values assigned to the *noise_floor*
        and *relative_error* properties.

        Where :math:`\varepsilon_{floor}` is the floor uncertainty and :math:`C_{err}`
        is the relative error for each datum :math:`d`, the total uncertainty
        :math:`\varepsilon` is given by:

        .. math::
            \varepsilon = \sqrt{\varepsilon_{floor}^2 + \big ( C_{err} |d| \big )^2}

        Returns
        -------
        numpy.ndarray
            The uncertainties applied to the data. Assuming the noise on the data
            are independent Gaussian with zero mean, the uncertainties represent
            estimates of the standard deviations of the noise on the data.
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
        """The number of observed data

        Returns
        -------
        int
            The number of observed data
        """
        return len(self.dobs)

    @property
    def shape(self):
        """The shape of the array containing the observed data

        Returns
        -------
        int or tuple of int
            The shape of the array containing the observed data
        """
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
        """Dictionary for indexing data by sources and receiver.

        FULL DESCRIPTION REQUIRED

        Returns
        -------
        dict
            Dictionary for indexing data by source and receiver

        Examples
        --------
        NEED EXAMPLE (1D TEM WOULD BE GOOD)

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
        """Convert observed data to a vector

        Returns
        -------
        (nD) numpy.ndarray
            Observed data organized in a vector
        """
        return self.dobs

    def fromvec(self, v):
        """Convert data to vector and assign to observed data

        This method converts the input numerical array *v* to a vector using
        `py:meth:mkvc`, then replaces the current *dobs* property of the data
        object with the new vector.

        Parameters
        ----------
        v : numpy.ndarray
            A numerical array that will be converted to a vector and
            used to replace the current quantity stored in the *dobs* property.
        """
        v = mkvc(v)
        self.dobs = v


class SyntheticData(Data):
    """Class for creating synthetic data.

    Parameters
    ----------
    survey : SimPEG.survey.BaseSurvey
        A SimPEG survey object. For each geophysical method, the survey object defines
        the survey geometry; i.e. sources, receivers, data type.
    dobs : numpy.ndarray
        Observed data.
    dclean : (nD) numpy.ndarray
        Noiseless data.
    relative_error : SimPEG.data.UncertaintyArray
        Assign relative uncertainties to the data using relative error; sometimes
        referred to as percent uncertainties. For each datum, we assume the
        standard deviation of Gaussian noise is the relative error times the
        absolute value of the datum; i.e. :math:`C_{err} \times |d|`.
    noise_floor : UncertaintyArray
        Assign floor/absolute uncertainties to the data. For each datum, we assume
        standard deviation of Gaussian noise is equal to *noise_floor*.
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
