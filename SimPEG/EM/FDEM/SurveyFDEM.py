import SimPEG
from SimPEG.EM.Utils import *
from scipy.constants import mu_0
from SimPEG.Utils import Zero, Identity
import SrcFDEM as Src


####################################################
# Receivers
####################################################

class Rx(SimPEG.Survey.BaseRx):

    knownRxTypes = {
                    'exr':['e', 'Ex', 'real'],
                    'eyr':['e', 'Ey', 'real'],
                    'ezr':['e', 'Ez', 'real'],
                    'exi':['e', 'Ex', 'imag'],
                    'eyi':['e', 'Ey', 'imag'],
                    'ezi':['e', 'Ez', 'imag'],

                    'bxr':['b', 'Fx', 'real'],
                    'byr':['b', 'Fy', 'real'],
                    'bzr':['b', 'Fz', 'real'],
                    'bxi':['b', 'Fx', 'imag'],
                    'byi':['b', 'Fy', 'imag'],
                    'bzi':['b', 'Fz', 'imag'],

                    'jxr':['j', 'Fx', 'real'],
                    'jyr':['j', 'Fy', 'real'],
                    'jzr':['j', 'Fz', 'real'],
                    'jxi':['j', 'Fx', 'imag'],
                    'jyi':['j', 'Fy', 'imag'],
                    'jzi':['j', 'Fz', 'imag'],

                    'hxr':['h', 'Ex', 'real'],
                    'hyr':['h', 'Ey', 'real'],
                    'hzr':['h', 'Ez', 'real'],
                    'hxi':['h', 'Ex', 'imag'],
                    'hyi':['h', 'Ey', 'imag'],
                    'hzi':['h', 'Ez', 'imag'],
                   }
    radius = None

    def __init__(self, locs, rxType):
        SimPEG.Survey.BaseRx.__init__(self, locs, rxType)

    @property
    def projField(self):
        """Field Type projection (e.g. e b ...)"""
        return self.knownRxTypes[self.rxType][0]

    @property
    def projGLoc(self):
        """Grid Location projection (e.g. Ex Fy ...)"""
        return self.knownRxTypes[self.rxType][1]

    @property
    def projComp(self):
        """Component projection (real/imag)"""
        return self.knownRxTypes[self.rxType][2]

    def projectFields(self, src, mesh, u):
        P = self.getP(mesh)
        u_part_complex = u[src, self.projField]
        # get the real or imag component
        real_or_imag = self.projComp
        u_part = getattr(u_part_complex, real_or_imag)
        return P*u_part

    def projectFieldsDeriv(self, src, mesh, u, v, adjoint=False):
        P = self.getP(mesh)

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

class Survey(SimPEG.Survey.BaseSurvey):
    """
        docstring for SurveyFDEM
    """

    srcPair = Src.BaseSrc

    def __init__(self, srcList, **kwargs):
        # Sort these by frequency
        self.srcList = srcList
        SimPEG.Survey.BaseSurvey.__init__(self, **kwargs)

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
        if getattr(self, '_nSrcByFreq', None) is None:
            self._nSrcByFreq = {}
            for freq in self.freqs:
                self._nSrcByFreq[freq] = len(self.getSrcByFreq(freq))
        return self._nSrcByFreq

    def getSrcByFreq(self, freq):
        """Returns the sources associated with a specific frequency."""
        assert freq in self._freqDict, "The requested frequency is not in this survey."
        return self._freqDict[freq]

    def projectFields(self, u):
        data = SimPEG.Survey.Data(self)
        for src in self.srcList:
            for rx in src.rxList:
                data[src, rx] = rx.projectFields(src, self.mesh, u)
        return data

    def projectFieldsDeriv(self, u):
        raise Exception('Use Sources to project fields deriv.')
