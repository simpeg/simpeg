import SimPEG
from SimPEG.EM.Utils import omega
from SimPEG.EM.Base import BaseEMSurvey
from scipy.constants import mu_0
from SimPEG.Utils import Zero, Identity
from . import SrcFDEM as Src
from . import RxFDEM as Rx


class Survey(BaseEMSurvey):
    """
    Frequency domain electromagnetic survey

    :param list srcList: list of FDEM sources used in the survey
    """

    srcPair = Src.BaseFDEMSrc
    rxPair = Rx.BaseRx

    def __init__(self, srcList, **kwargs):
        # Sort these by frequency
        self.srcList = srcList
        BaseEMSurvey.__init__(self, srcList, **kwargs)

        _freqDict = {}
        for src in srcList:
            if src.freq not in _freqDict:
                _freqDict[src.freq] = []
            _freqDict[src.freq] += [src]

        self._freqDict = _freqDict
        self._freqs = sorted([f for f in self._freqDict])

    @property
    def freqs(self):
        """Frequencies"""
        return self._freqs

    @property
    def nFreq(self):
        """Number of frequencies"""
        return len(self._freqDict)

    @property
    def nSrcByFreq(self):
        """Number of sources at each frequency"""
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

