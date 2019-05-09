import properties
import numpy as np
import warnings

from .survey import BaseSurvey
from .utils import mkvc

__all__ = ['Data', 'SyntheticData']


class BaseDataArray(properties.HasProperties):

    class_info = "Base class for a data array that can be indexed by a source, receiver pair"

    data = properties.Array(
        "data that can be indexed by a source receiver pair",
        shape=('*',),
        required=True,
    )

    survey = properties.Instance(
        "a SimPEG survey object",
        BaseSurvey
    )

    def __init__(self, survey, data=None):
        super(BaseDataArray, self).__init__()
        self.survey = survey
        if data is None:
            data = np.empty(survey.nD)
        self.data = data

    def _set_data_dict(self):
        if self.survey is None:
            raise Exception(
                "To set or get values by source-receiver pairs, a survey must "
                "first be set. `data.survey = survey`"
            )

        # create an empty dict
        self._data_dict = {}

        # create an empty dict associated with each source
        for src in self.survey.source_list:
            self._data_dict[src] = {}

        # loop over sources and find the associated data indices
        indBot, indTop = 0, 0
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                indTop += rx.nD
                self._data_dict[src][rx] = np.arange(indBot, indTop)
                indBot += rx.nD


    def _ensureCorrectKey(self, key):
        if type(key) is tuple:
            if len(key) is not 2:
                raise KeyError('Key must be [Src, Rx]')
            if key[0] not in self.survey.source_list:
                raise KeyError('Src Key must be a source in the survey.')
            if key[1] not in key[0].receiver_list:
                raise KeyError('Rx Key must be a receiver for the source.')
            return key

        elif key not in self.survey.source_list:
            raise KeyError('Key must be a source in the survey.')
        return key, None

    def __setitem__(self, key, value):
        "set item"
        src, rx = self._ensureCorrectKey(key)
        assert rx is not None, 'set data using [Src, Rx]'
        assert isinstance(value, np.ndarray), 'value must by ndarray'
        assert value.size == rx.nD, (
            "value must have the same number of data as the source."
        )

        if getattr(self, '_data_dict', None) is None:
            self._set_data_dict()

        inds = self._data_dict[src][rx]
        self.data[inds] = mkvc(value)

    def __getitem__(self, key):
        src, rx = self._ensureCorrectKey(key)

        if getattr(self, '_data_dict', None) is None:
            self._set_data_dict()

        if rx is not None:
            if rx not in self._data_dict[src]:
                raise Exception('Data for receiver has not yet been set.')
            return self.data[self._data_dict[src][rx]]

        return np.concatenate([
            self.data[self._data_dict[src][rx]] for rx in src.receiver_list
        ])

    @property
    def nD(self):
        return len(self.data)

    def tovec(self):
        if len(self.survey.source_list) == 0:
            return self.data
        return np.concatenate([self[src] for src in self.survey.source_list])

    def fromvec(self, v):
        v = utils.mkvc(v)
        assert v.size == self.survey.nD, (
            'v must have the correct number of data.'
        )
        indBot, indTop = 0, 0
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                indTop += rx.nD
                self[src, rx] = v[indBot:indTop]
                indBot += rx.nD


class Data(properties.HasProperties):
    """
    Data storage
    """

    _dobs = properties.Instance(
        "Observed data",
        BaseDataArray,
        required=True
    )

    _standard_deviation = properties.Instance(
        "standard deviation of the data",
        BaseDataArray,
        required=True
    )

    _noise_floor = properties.Instance(
        "noise floor of the data",
        BaseDataArray,
        required=True
    )

    survey = properties.Instance(
        "a SimPEG survey object",
        BaseSurvey
    )

    _uid = properties.Uuid("unique ID for the data")

    def __init__(
        self, survey, dobs=None, standard_deviation=None, noise_floor=None
    ):
        super(Data, self).__init__()
        self.survey = survey

        # Observed data
        self._dobs = BaseDataArray(survey=survey)
        if dobs is not None:
            self._dobs.data = dobs
            self._dobs._set_data_dict()

        # Standard deviation (use the data dict from the observed data)
        self._standard_deviation = BaseDataArray(survey=survey)
        if standard_deviation is not None:
            self.standard_deviation = standard_deviation  # go through the setter

        # Noise floor (use the data dict from the observed data)
        self._noise_floor = BaseDataArray(survey=survey)
        if noise_floor is not None:
            self.noise_floor = noise_floor  # go through the setter


    @property
    def dobs(self):
        return self._dobs.tovec()

    @dobs.setter
    def dobs(self, value):
        if isinstance(value, BaseDataArray):
            self._dobs = value
        else:
            self._dobs.data = value
        self._dobs._set_data_dict()

    @property
    def standard_deviation(self):
        return self._standard_deviation.tovec()

    @standard_deviation.setter
    def standard_deviation(self, value):
        if isinstance(value, BaseDataArray):
            self._standard_deviation = value
        else:
            if isinstance(value, float):  # set this to a vector the same length as the data
                value = value * np.abs(self.dobs)
            self._standard_deviation = BaseDataArray(data=value, survey=self.survey)
        if getattr(self.dobs, '_data_dict', None) is not None:  # skip creating the data_dict and assign it
            self._standard_deviation._data_dict = self._dobs._data_dict

    @property
    def noise_floor(self):
        return self._noise_floor.tovec()

    @noise_floor.setter
    def noise_floor(self, value):
        if isinstance(value, BaseDataArray):
            self._noise_floor = value
        elif isinstance(value, float):
            value = value * np.ones(self.nD)  # set this to a vector the same length as the data
        self._noise_floor = BaseDataArray(data=value, survey=self.survey)
        if getattr(self, 'dobs', None) is not None: # skip creating the data_dict and assign it
            self._noise_floor._data_dict = self._dobs._data_dict


    @property
    def uncertainty(self):
        """
        Data uncertainties. If a stardard deviation and noise floor are
        provided, the incertainty is

        ..code:: python

            data.uncertainty == (
                data.standard_deviation * np.absolute(data.dobs) +
                data.noise_floor
            )

        otherwise, the uncertainty can be set directly

        ..code:: python

            data.uncertainty = 0.05 * np.absolute(self.dobs) + 1e-12

        """
        if self.standard_deviation is None and self.noise_floor is None:
            raise Exception(
                "The standard_deviation and / or noise_floor must be set "
                "before asking for uncertainties. Alternatively, the "
                "uncertainty can be set directly"
            )

        uncert = np.zeros(self.nD)
        if self.standard_deviation is not None:
            uncert = uncert + self.standard_deviation * np.absolute(self.dobs)
        if self.noise_floor is not None:
            uncert = uncert + self.noise_floor

        return uncert

    @uncertainty.setter
    def uncertainty(self, value):
        del self.standard_deviation
        self.noise_floor = value

    @property
    def nD(self):
        return self._dobs.nD

    def __setitem__(self, key, value):
        return self._dobs.__setitem__(key, value)

    def __getitem__(self, key):
        return self._dobs.__getitem__(key)

    def tovec(self):
        return self._dobs.tovec()

    def fromvec(self, v):
        return self._dobs.fromvec(v)

    ##########################
    # Depreciated properties #
    ##########################
    @property
    def std(self):
        warnings.warn(
            "std has been depreciated in favor of standard_deviation. Please "
            "update your code to use 'standard_deviation'", DeprecationWarning
        )
        return self.standard_deviation

    @std.setter
    def std(self, value):
        warnings.warn(
            "std has been depreciated in favor of standard_deviation. Please "
            "update your code to use 'standard_deviation'", DeprecationWarning
        )
        self.standard_deviation = value

    @property
    def eps(self):
        warnings.warn(
            "eps has been depreciated in favor of noise_floor. Please "
            "update your code to use 'noise_floor'", DeprecationWarning
        )
        return self.noise_floor

    @eps.setter
    def eps(self, value):
        warnings.warn(
            "eps has been depreciated in favor of noise_floor. Please "
            "update your code to use 'noise_floor'", DeprecationWarning
        )
        self.noise_floor = value


class SyntheticData(Data):
    """
    Data class for synthetic data. It keeps track of observed and clean data
    """
    _dclean = properties.Instance(
        "Observed data",
        BaseDataArray,
        required=True
    )

    def __init__(self, survey, dobs=None, dclean=None, standard_deviation=None, noise_floor=None):
        super(SyntheticData, self).__init__(
            survey=survey, dobs=dobs,
            standard_deviation=standard_deviation, noise_floor=noise_floor
        )
        self._dclean = BaseDataArray(survey=survey)
        if dclean is not None:
            self.dclean = dclean

    @property
    def dclean(self):
        return self._dclean.data.tovec()

    @dclean.setter
    def dclean(self, value):
        if isinstance(value, BaseDataArray):
            self._dclean = value
        else:
            if isinstance(value, float):  # set this to a vector the same length as the data
                value = value * np.abs(self.dobs)
            self._dclean = BaseDataArray(data=value, survey=self.survey)
        if getattr(self._dobs, '_data_dict', None) is not None:  # skip creating the data_dict and assign it
            self._dclean._data_dict = self._dobs._data_dict
