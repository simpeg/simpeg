from scipy.constants import mu_0
import properties

from ...utils import Zero, Identity
from ..Utils import omega
from ...survey import BaseSurvey
from .source import BaseFDEMSrc

import warnings


class Survey(BaseSurvey):
    """
    Frequency domain electromagnetic survey
    """

    source_list = properties.List(
        "A list of sources for the survey",
        properties.Instance("A SimPEG source", BaseFDEMSrc),
        default=[]
    )

    def __init__(self, source_list=None, **kwargs):
        # Sort these by frequency
        super(Survey, self).__init__(source_list, **kwargs)

        _frequency_dict = {}
        for src in source_list:
            if src.freq not in _frequency_dict:
                _frequency_dict[src.freq] = []
            _frequency_dict[src.freq] += [src]

        self._frequency_dict = _frequency_dict
        self._frequencies = sorted([f for f in self._frequency_dict])

    @property
    def frequencies(self):
        """
        Frequencies in the survey
        """
        return self._frequencies

    @property
    def freqs(self):
        """Frequencies"""
        warnings.warn(
            "survey.freqs will be depreciated in favor of survey.frequencies. "
            "Please update your code accordingly."
        )
        return self.frequencies

    @property
    def num_frequencies(self):
        """Number of frequencies"""
        return len(self._frequency_dict)

    @property
    def nFreq(self):
        """Number of frequencies"""
        warnings.warn(
            "survey.nFreq will be depreciated in favor of survey.num_frequencies. "
            "Please update your code accordingly."
        )
        return self.num_frequencies

    @property
    def num_sources_by_frequency(self):
        """Number of sources at each frequency"""
        if getattr(self, '_num_sources_by_frequency', None) is None:
            self._num_sources_by_frequency = {}
            for freq in self.frequencies:
                self._num_sources_by_frequency[freq] = len(self.getSrcByFreq(freq))
        return self._num_sources_by_frequency

    @property
    def nSrcByFreq(self):
        """Number of sources at each frequency"""
        warnings.warn(
            "survey.nSrcByFreq will be depreciated in favor of survey.num_sources_by_frequency. "
            "Please update your code accordingly."
        )
        return self.num_sources_by_frequency

    def get_sources_by_frequency(self, frequency):
        """
        Returns the sources associated with a specific frequency.
        :param float frequency: frequency for which we look up sources
        :rtype: dictionary
        :return: sources at the sepcified frequency
        """
        assert frequency in self._frequency_dict, (
            "The requested frequency is not in this survey."
        )
        return self._frequency_dict[frequency]

    def getSrcByFreq(self, frequency):
        """
        Returns the sources associated with a specific frequency.
        :param float frequency: frequency for which we look up sources
        :rtype: dictionary
        :return: sources at the sepcified frequency
        """
        warnings.warn(
            "survey.getSrcByFreq will be depreciated in favor of survey.get_sources_by_frequency. "
            "Please update your code accordingly."
        )
        return self.get_sources_by_frequency(frequency)

