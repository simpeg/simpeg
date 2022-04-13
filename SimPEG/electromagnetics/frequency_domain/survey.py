from scipy.constants import mu_0
# import properties
from ...utils.code_utils import deprecate_property, deprecate_method

from ...utils import Zero, Identity
from ..utils import omega
from ...survey import BaseSurvey
from .sources import BaseFDEMSrc


class Survey(BaseSurvey):
    """Frequency domain electromagnetic survey

    Parameters
    ----------
    source_list : list of SimPEG.electromagnetic.frequency_domain.sources.BaseFDEMSrc
        List of SimPEG FDEM sources
    """

    # source_list = properties.List(
    #     "A list of sources for the survey",
    #     properties.Instance("A SimPEG source", BaseFDEMSrc),
    #     default=[],
    # )

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
    def source_list(self):
        """List of FDEM sources associated with the survey

        Returns
        -------
        list of BaseFDEMSrc
            List of FDEM sources associated with the survey
        """
        return self._source_list

    @source_list.setter
    def source_list(self, new_list):
        if not isinstance(new_list, list):
            new_list = [new_list]
        
        if any([isinstance(x, BaseFDEMSrc)==False for x in new_list]):
            raise TypeError("Source list must be a list of SimPEG.survey.BaseFDEMSrc")

        assert len(set(new_list)) == len(new_list), "The source_list must be unique. Cannot re-use sources"

        self._sourceOrder = dict()
        [self._sourceOrder.setdefault(src._uid, ii) for ii, src in enumerate(new_list)]
        self._source_list = new_list

    @property
    def frequencies(self):
        """Frequencies in the survey

        Returns
        -------
        int
            Frequencies used in the survey
        """
        return self._frequencies

    freqs = deprecate_property(
        frequencies,
        "freq",
        new_name="frequencies",
        removal_version="0.16.0",
        error=True,
    )

    @property
    def num_frequencies(self):
        """Number of frequencies

        Returns
        -------
        int
            Number of frequencies
        """
        return len(self._frequency_dict)

    nFreq = deprecate_property(
        num_frequencies,
        "nFreq",
        new_name="num_frequencies",
        removal_version="0.16.0",
        error=True,
    )

    @property
    def num_sources_by_frequency(self):
        """Number of sources at each frequency

        Returns
        -------
        list of int
            Number of sources associated with each frequency
        """
        if getattr(self, "_num_sources_by_frequency", None) is None:
            self._num_sources_by_frequency = {}
            for freq in self.frequencies:
                self._num_sources_by_frequency[freq] = len(
                    self.get_sources_by_frequency(freq)
                )
        return self._num_sources_by_frequency

    nSrcByFreq = deprecate_property(
        num_sources_by_frequency,
        "nSrcByFreq",
        new_name="num_sources_by_frequency",
        removal_version="0.16.0",
        error=True,
    )

    def get_sources_by_frequency(self, frequency):
        """Get sources by frequency

        Parameters
        ----------
        frequency : float
            Frequency

        Returns
        -------
        dict
            sources at the sepcified frequency
        """
        assert (
            frequency in self._frequency_dict
        ), "The requested frequency is not in this survey."
        return self._frequency_dict[frequency]

    getSrcByFreq = deprecate_method(
        get_sources_by_frequency, "getSrcByFreq", "0.16.0", error=True
    )
