import numpy as np
import warnings

from .survey import BaseSurvey
from .utils import mkvc, validate_ndarray_with_shape, validate_float, validate_type

__all__ = ["Data", "SyntheticData"]


class Data:
    r"""Class for defining data in SimPEG.

    The ``Data`` class is used to create an object which connects the survey geometry,
    observed data and data uncertainties.

    Parameters
    ----------
    survey : simpeg.survey.BaseSurvey
        A SimPEG survey object. For each geophysical method, the survey object defines
        the survey geometry; i.e. sources, receivers, data type.
    dobs : (n) numpy.ndarray
        Observed data.
    relative_error : None or float or numpy.ndarray, optional
        Assign relative uncertainties to the data using relative error; sometimes
        referred to as percent uncertainties. For each datum, we assume the
        standard deviation of Gaussian noise is the relative error times the
        absolute value of the datum; i.e. :math:`C_{err} \times |d|`.
    noise_floor : None or float or numpy.ndarray, optional
        Assign floor/absolute uncertainties to the data. For each datum, we assume
        standard deviation of Gaussian noise is equal to *noise_floor*.
    standard_deviation : None or float or numpy.ndarray, optional
        Directly define the uncertainties on the data by assuming we know the standard
        deviations of the Gaussian noise. This is essentially the same as *noise_floor*.
        If set however, this will override *relative_error* and *noise_floor*. If none
        are given, this defaults to 0.0

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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.survey = survey

        # Observed data
        if dobs is None:
            dobs = np.full(survey.nD, np.nan)  # initialize data as nans
        self.dobs = dobs

        self.relative_error = relative_error

        self.noise_floor = noise_floor

        if standard_deviation is not None:
            if relative_error is not None or noise_floor is not None:
                warnings.warn(
                    "Setting the standard_deviation overwrites the "
                    "relative_error and noise_floor",
                    stacklevel=2,
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
    def survey(self):
        """The survey for this data.

        Returns
        -------
        simpeg.simulation.BaseSurvey
        """
        return self._survey

    @survey.setter
    def survey(self, value):
        self._survey = validate_type("survey", value, BaseSurvey, cast=False)

    @property
    def dobs(self):
        """Vector of the observed data.

        Returns
        -------
        numpy.ndarray

        Notes
        -----
        This array can also be modified by directly indexing the data object
        using the a tuple of the survey's sources and receivers.

        >>> data = Data(survey)
        >>> for src in survey.source_list:
        ...     for rx in src.receiver_list:
        ...         data[src, rx] = datum
        """
        return self._dobs

    @dobs.setter
    def dobs(self, value):
        self._dobs = validate_ndarray_with_shape(
            "dobs", value, shape=(self.survey.nD,), dtype=(float, complex)
        )

    @property
    def relative_error(self):
        """Relative error of the data.

        This can be set using an array of the
        same size as the data (e.g. if you want to assign a different relative
        error to each datum) or as a scalar if you would like to assign a
        the same relative error to all data.

        The standard_deviation is constructed as follows::

            np.sqrt( (relative_error * np.abs(dobs))**2 + noise_floor**2 )

        For example, if you set

        >>> data = Data(survey, dobs=dobs)
        >>> data.relative_error = 0.05

        then the contribution to the standard_deviation is equal to

        >>> data.relative_error * np.abs(data.dobs)

        Returns
        -------
        None or float or numpy.ndarray
        """
        return self._relative_error

    @relative_error.setter
    def relative_error(self, value):
        if value is not None:
            try:
                value = validate_float("relative_error", value)
                value = np.full(self.survey.nD, value)
            except TypeError:
                pass
            value = validate_ndarray_with_shape(
                "relative_error", value, shape=(self.survey.nD,)
            )
            if np.any(value < 0.0):
                raise ValueError("relative_error must be positive.")
        self._relative_error = value

    @property
    def noise_floor(self):
        """Noise floor of the data.

        This can be set using an array of the
        same size as the data (e.g. if you want to assign a different noise
        floor to each datum) or as a scalar if you would like to assign a
        the same noise floor to all data.

        The standard_deviation is constructed as follows::

            np.sqrt( (relative_error * np.abs(dobs))**2 + noise_floor**2 )

        For example, if you set

        >>> data = Data(survey, dobs=dobs)
        >>> data.noise_floor = 1e-10

        then the contribution to the standard_deviation is equal to

        >>> data.noise_floor

        Returns
        -------
        None or float or numpy.ndarray
        """
        return self._noise_floor

    @noise_floor.setter
    def noise_floor(self, value):
        if value is not None:
            try:
                value = validate_float("noise_floor", value)
                value = np.full(self.survey.nD, value)
            except TypeError:
                pass
            value = validate_ndarray_with_shape(
                "noise_floor", value, shape=(self.survey.nD,)
            )
            if np.any(value < 0.0):
                raise ValueError("noise_floor must be positive.")
        self._noise_floor = value

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
            raise TypeError(
                "The relative_error and / or noise_floor must be set "
                "before asking for uncertainties. Alternatively, the "
                "standard_deviation can be set directly"
            )

        uncert = np.zeros(self.nD)
        if self.relative_error is not None:
            uncert += (self.relative_error * np.absolute(self.dobs)) ** 2
        if self.noise_floor is not None:
            uncert += self.noise_floor**2

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
    r"""Synthetic data class.

    The ``SyntheticData`` class is a :py:class:`simpeg.data.Data` class that allows the
    user to keep track of both clean and noisy data.

    Parameters
    ----------
    survey : simpeg.survey.BaseSurvey
        A SimPEG survey object. For each geophysical method, the survey object defines
        the survey geometry; i.e. sources, receivers, data type.
    dobs : numpy.ndarray
        Observed data.
    dclean : (nD) numpy.ndarray
        Noiseless data.
    relative_error : float or np.ndarray
        Assign relative uncertainties to the data using relative error; sometimes
        referred to as percent uncertainties. For each datum, we assume the
        standard deviation of Gaussian noise is the relative error times the
        absolute value of the datum; i.e. :math:`C_{err} \times |d|`.
    noise_floor : float or np.ndarray
        Assign floor/absolute uncertainties to the data. For each datum, we assume
        standard deviation of Gaussian noise is equal to *noise_floor*.
    """

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
            dclean = np.full(self.survey.nD, np.nan)
        self.dclean = dclean

    @property
    def dclean(self):
        """
        Vector of the clean synthetic data.

        Returns
        -------
        numpy.ndarray

        Notes
        --------
        This array should be indexing the data object
        using the a tuple of the survey's sources and receivers.

        >>> data = Data(survey)
        >>> for src in survey.source_list:
        ...     for rx in src.receiver_list:
        ...         index = data.index_dictionary(src, rx)
        ...         data.dclean[src, rx] = datum

        """
        return self._dclean

    @dclean.setter
    def dclean(self, value):
        self._dclean = validate_ndarray_with_shape(
            "dclean", value, shape=(self.survey.nD,), dtype=(float, complex)
        )
