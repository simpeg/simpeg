import properties
from ...Utils import Zero
from ...NewSurvey import BaseSurvey
from ..NewBase import BaseEMSrc
from .RxFDEM import BaseFDEMRx


###############################################################################
#                                                                             #
#                            Base FDEM Source                                 #
#                                                                             #
###############################################################################

class BaseFDEMSrc(BaseEMSrc):
    """
    Base source class for FDEM Survey. Inherit this to build your own FDEM
    source.
    """

    freq = properties.Float(
        "frequency of the source",
        min=0, required=True
    )

    rxList = properties.List(
        "list of FDEM receivers",
        properties.Instance("FDEM receiver", BaseFDEMRx),
        default=[]
    )

    def __init__(self, **kwargs):
        # TODO: spell out frequency
        # freq = kwargs.pop('freq', None)
        # if freq is not None:
        #     warnings.warn(
        #         "the keyword argument 'freq' will be depreciated in favour of "
        #         "'frequency' please use src(frequency={}) to create the "
        #         "source".format(freq)
        #     )
        #     kwargs['frequency'] = freq
        super(BaseFDEMSrc, self).__init__(**kwargs)

    def bPrimary(self, simulation):
        """
        Primary magnetic flux density

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :rtype: numpy.ndarray
        :return: primary magnetic flux density
        """
        if getattr(self, '_bPrimary', None) is None:
            self._bPrimary = Zero()
        return self._bPrimary

    def bPrimaryDeriv(self, simulation, v, adjoint=False):
        """
        Derivative of the primary magnetic flux density

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :param numpy.ndarray v: vector
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: primary magnetic flux density
        """
        return Zero()

    def hPrimary(self, simulation):
        """
        Primary magnetic field

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        if getattr(self, '_hPrimary', None) is None:
            self._hPrimary = Zero()
        return self._hPrimary

    def hPrimaryDeriv(self, simulation, v, adjoint=False):
        """
        Derivative of the primary magnetic field

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :param numpy.ndarray v: vector
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: primary magnetic flux density
        """
        return Zero()

    def ePrimary(self, simulation):
        """
        Primary electric field

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :rtype: numpy.ndarray
        :return: primary electric field
        """
        if getattr(self, '_ePrimary', None) is None:
            self._ePrimary = Zero()
        return self._ePrimary

    def ePrimaryDeriv(self, simulation, v, adjoint=False):
        """
        Derivative of the primary electric field

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :param numpy.ndarray v: vector
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: primary magnetic flux density
        """
        return Zero()

    def jPrimary(self, simulation):
        """
        Primary current density

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :rtype: numpy.ndarray
        :return: primary current density
        """
        if getattr(self, '_jPrimary', None) is None:
            self._jPrimary = Zero()
        return self._jPrimary

    def jPrimaryDeriv(self, simulation, v, adjoint=False):
        """
        Derivative of the primary current density

        :param BaseFDEMSimulation simulation: FDEM Simulation
        :param numpy.ndarray v: vector
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: primary magnetic flux density
        """
        return Zero()


###############################################################################
#                                                                             #
#                                  Survey                                     #
#                                                                             #
###############################################################################

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
        if getattr(self, '_freqs', None) is None:
            _freqDict = {}

            for src in self.srcList:
                if src.freq not in _freqDict:
                    _freqDict[src.freq] = []
                _freqDict[src.freq] += [src]

            self._freqDict = _freqDict
            self._freqs = sorted([f for f in self._freqDict])
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
