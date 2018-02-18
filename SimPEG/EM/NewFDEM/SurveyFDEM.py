import properties
from ...Survey import BaseSurvey
from .SrcFDEM import BaseFDEMSrc


class Survey(BaseSurvey):
    """
    A frequency domain electromagnetic survey
    """

    srcList = properties.List(
        "A list of sources for the survey",
        properties.Instance(
            "A frequency domain EM source",
            BaseFDEMSrc
        ),
        required=True
    )

    def __init__(self, **kwargs):
        super(Survey, self).__init__(**kwargs)

    @properties.observer('srcList')
    def _set_freqs(self, change):
        srcList = change['value']
        _freqDict = {}

        for src in srcList:
            if src.freq not in _freqDict:
                _freqDict[src.freq] = []
            _freqDict[src.freq] += [src]

        self._freqDict = _freqDict
        self._freqs = sorted([f for f in self._freqDict])

    # @property
    # def frequencies(self):
    #     """
    #     Frequencies in the FDEM survey
    #     """
    #     return self._freqs

    @property
    def freqs(self):
        """
        Frequencies in the FDEM survey
        """
        return self._freqs

    @property
    def nFreq(self):
        """
        Number of frequencies in the survey
        """
        return len(self._freqDict)

    @property
    def nSrcByFreq(self):
        """
        Number of sources at each frequency
        """
        if getattr(self, '_nSrcByFreq', None) is None:
            self._nSrcByFreq = {}
            for freq in self.freqs:
                self._nSrcByFreq[freq] = len(self.getSrcByFreq(freq))
        return self._nSrcByFreq

    def getSrcByFreq(self, freq):
        """
        Returns the sources associated with a specific frequency.
        :param float freq: frequency for which we look up sources
        :rtype: dictionary
        :return: sources at the sepcified frequency
        """
        assert freq in self._freqDict, (
            "The requested frequency is not in this survey."
        )
        return self._freqDict[freq]
