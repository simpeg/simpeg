import SimPEG
from SimPEG.EM.Utils import *
from SimPEG.EM.Base import BaseEMSurvey
from scipy.constants import mu_0
from SimPEG.Utils import Zero, Identity
import SrcFDEM as Src
from SimPEG import sp


####################################################
# Receivers
####################################################

class Rx(SimPEG.Survey.BaseRx):
    """
    Frequency domain receivers

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string rxType: reciever type from knownRxTypes
    """

    knownRxTypes = {
                    'exr':['e', 'x', 'real'],
                    'eyr':['e', 'y', 'real'],
                    'ezr':['e', 'z', 'real'],
                    'exi':['e', 'x', 'imag'],
                    'eyi':['e', 'y', 'imag'],
                    'ezi':['e', 'z', 'imag'],

                    'bxr':['b', 'x', 'real'],
                    'byr':['b', 'y', 'real'],
                    'bzr':['b', 'z', 'real'],
                    'bxi':['b', 'x', 'imag'],
                    'byi':['b', 'y', 'imag'],
                    'bzi':['b', 'z', 'imag'],

                    'jxr':['j', 'x', 'real'],
                    'jyr':['j', 'y', 'real'],
                    'jzr':['j', 'z', 'real'],
                    'jxi':['j', 'x', 'imag'],
                    'jyi':['j', 'y', 'imag'],
                    'jzi':['j', 'z', 'imag'],

                    'hxr':['h', 'x', 'real'],
                    'hyr':['h', 'y', 'real'],
                    'hzr':['h', 'z', 'real'],
                    'hxi':['h', 'x', 'imag'],
                    'hyi':['h', 'y', 'imag'],
                    'hzi':['h', 'z', 'imag'],
                   }
    radius = None

    def __init__(self, locs, rxType):
        SimPEG.Survey.BaseRx.__init__(self, locs, rxType)

    @property
    def projField(self):
        """Field Type projection (e.g. e b ...)"""
        return self.knownRxTypes[self.rxType][0]

    @property
    def projComp(self):
        """Component projection (real/imag)"""
        return self.knownRxTypes[self.rxType][2]

    def projGLoc(self, u):
        """Grid Location projection (e.g. Ex Fy ...)"""
        return u._GLoc(self.rxType[0]) + self.knownRxTypes[self.rxType][1]

    def eval(self, src, mesh, f):
        """
        Project fields to recievers to get data.

        :param Source src: FDEM source
        :param Mesh mesh: mesh used
        :param Fields f: fields object
        :rtype: numpy.ndarray
        :return: fields projected to recievers
        """
        # projGLoc = u._GLoc(self.knownRxTypes[self.rxType][0])
        # projGLoc += self.knownRxTypes[self.rxType][1]

        P = self.getP(mesh, self.projGLoc(f))
        f_part_complex = f[src, self.projField]
        # get the real or imag component
        real_or_imag = self.projComp
        f_part = getattr(f_part_complex, real_or_imag)

        return P*f_part

    def evalDeriv(self, src, mesh, f, v, adjoint=False):
        """
        Derivative of projected fields with respect to the inversion model times a vector.

        :param Source src: FDEM source
        :param Mesh mesh: mesh used
        :param Fields f: fields object
        :param numpy.ndarray v: vector to multiply
        :rtype: numpy.ndarray
        :return: fields projected to recievers
        """

        P = self.getP(mesh, self.projGLoc(f))

        if not adjoint:
            Pv_complex = P * v
            real_or_imag = self.projComp
            Pv = getattr(Pv_complex, real_or_imag)
        elif adjoint:
            Pv_real = P.T * v

            real_or_imag = self.projComp
            if real_or_imag == 'imag':
                Pv = 1j*Pv_real
            elif real_or_imag == 'real':
                Pv = Pv_real.astype(complex)
            else:
                raise NotImplementedError('must be real or imag')

        return Pv


####################################################
# Survey
####################################################

class Survey(BaseEMSurvey):
    """
    Frequency domain electromagnetic survey

    :param list srcList: list of FDEM sources used in the survey
    """

    srcPair = Src.BaseSrc
    rxPair = Rx

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
        assert freq in self._freqDict, "The requested frequency is not in this survey."
        return self._freqDict[freq]

