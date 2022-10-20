from scipy.constants import mu_0
from ...survey import BaseSurvey
from .sources import BaseFDEMSrc
from ...utils import validate_list_of_types


class Survey(BaseSurvey):
    """Frequency domain electromagnetic survey

    Parameters
    ----------
    source_list : list of SimPEG.electromagnetic.frequency_domain.sources.BaseFDEMSrc
        List of SimPEG FDEM sources
    """

    def __init__(self, source_list, **kwargs):

        super(Survey, self).__init__(source_list, **kwargs)

        _frequency_dict = {}
        for src in self.source_list:
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
        self._source_list = validate_list_of_types(
            "source_list", new_list, BaseFDEMSrc, ensure_unique=True
        )

    @property
    def frequencies(self):
        """Frequencies in the survey

        Returns
        -------
        int
            Frequencies used in the survey
        """
        return self._frequencies

    @property
    def num_frequencies(self):
        """Number of frequencies

        Returns
        -------
        int
            Number of frequencies
        """
        return len(self._frequency_dict)

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
