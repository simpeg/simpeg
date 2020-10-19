from scipy.constants import mu_0
import properties
from ...utils.code_utils import deprecate_property, deprecate_method

from ...utils import Zero, Identity
from ..utils import omega
from ...survey import BaseSurvey
from .sources import BaseFDEMSrc


class Survey(BaseSurvey):
    """
    Frequency domain electromagnetic survey
    """

    source_list = properties.List(
        "A list of sources for the survey",
        properties.Instance("A SimPEG source", BaseFDEMSrc),
        default=[],
    )

    def __init__(self, source_list=None, **kwargs):
        # Sort these by frequency
        super(Survey, self).__init__(source_list, **kwargs)

        _frequency_dict = {}
        for src in source_list:
            if src.frequency not in _frequency_dict:
                _frequency_dict[src.frequency] = []
            _frequency_dict[src.frequency] += [src]

        self._frequency_dict = _frequency_dict
        self._frequencies = sorted([f for f in self._frequency_dict])

    @property
    def frequencies(self):
        """
        Frequencies in the survey
        """
        return self._frequencies

    freqs = deprecate_property(
        frequencies, "freq", new_name="frequencies", removal_version="0.15.0"
    )

    @property
    def num_frequencies(self):
        """Number of frequencies"""
        return len(self._frequency_dict)

    nFreq = deprecate_property(
        num_frequencies, "nFreq", new_name="num_frequencies", removal_version="0.15.0"
    )

    @property
    def num_sources_by_frequency(self):
        """Number of sources at each frequency"""
        if getattr(self, "_num_sources_by_frequency", None) is None:
            self._num_sources_by_frequency = {}
            for freq in self.frequencies:
                self._num_sources_by_frequency[freq] = len(self.getSrcByFreq(freq))
        return self._num_sources_by_frequency

    nSrcByFreq = deprecate_property(
        num_sources_by_frequency,
        "nSrcByFreq",
        new_name="num_sources_by_frequency",
        removal_version="0.15.0",
    )

    def get_sources_by_frequency(self, frequency):
        """
        Returns the sources associated with a specific frequency.
        :param float frequency: frequency for which we look up sources
        :rtype: dictionary
        :return: sources at the sepcified frequency
        """
        assert (
            frequency in self._frequency_dict
        ), "The requested frequency is not in this survey."
        return self._frequency_dict[frequency]

    getSrcByFreq = deprecate_method(get_sources_by_frequency, "getSrcByFreq", "0.15.0")
