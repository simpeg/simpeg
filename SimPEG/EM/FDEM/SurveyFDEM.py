import SimPEG
from SimPEG.EM.Utils import *
from scipy.constants import mu_0
from SimPEG.Utils import Zero, Identity
import SrcFDEM as Src
from SimPEG import sp


####################################################
# Receivers
####################################################

class Rx(SimPEG.Survey.BaseRx):

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
        print 'here', u._GLoc(self.rxType[0]) + self.knownRxTypes[self.rxType][1]
        return u._GLoc(self.rxType[0]) + self.knownRxTypes[self.rxType][1]

    def projectFields(self, src, mesh, u):

        # projGLoc = u._GLoc(self.knownRxTypes[self.rxType][0])
        # projGLoc += self.knownRxTypes[self.rxType][1]

        P = self.getP(mesh, self.projGLoc(u))

        u_part_complex = u[src, self.projField]
        # get the real or imag component
        real_or_imag = self.projComp
        u_part = getattr(u_part_complex, real_or_imag)

        

        # if projGLoc == 'CC':
        #     P = self.getP(mesh, projGLoc)
        #     Z = 0.*P
        #     if mesh.dim == 3:
        #         if mesh._meshType == 'CYL' and mesh.isSymmetric and u_part.size > mesh.nC: # TODO: there must be a better way to do this!
        #             if self.knownRxTypes[self.rxType][1] == 'x':
        #                 P = sp.hstack([P,Z])
        #             elif self.knownRxTypes[self.rxType][1] == 'z':
        #                 P = sp.hstack([Z,P])
        #             elif self.knownRxTypes[self.rxType][1] == 'y':
        #                 raise Exception('Symmetric CylMesh does not support y interpolation, as this variable does not exist.')
        #         else:
        #             if self.knownRxTypes[self.rxType][1] == 'x':
        #                 P = sp.hstack([P,Z,Z])
        #             elif self.knownRxTypes[self.rxType][1] == 'y':
        #                 P = sp.hstack([Z,P,Z])
        #             elif self.knownRxTypes[self.rxType][1] == 'z':
        #                 P = sp.hstack([Z,Z,P])
        # else: 
        #     projGLoc += self.knownRxTypes[self.rxType][1]
        #     P = self.getP(mesh, projGLoc)
        
        return P*u_part

    def projectFieldsDeriv(self, src, mesh, u, v, adjoint=False):

        projGLoc = u._GLoc(self.knownRxTypes[self.rxType][0])

        print self.knownRxTypes[self.rxType][:2], 'Deriv', projGLoc
        projGLoc += self.knownRxTypes[self.rxType][1]

        # if projGLoc = 'CC':
        #     P = self.getP(mesh)
        #     if sel

        # else projGLoc != 'CC': 
        # projGLoc += self.knownRxTypes[self.rxType][1]
        P = self.getP(mesh)

        if not adjoint:
            print P.shape, v.shape
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
                print rx.nD
                dat = rx.projectFields(src, self.mesh, u)
                print dat.shape
                data[src, rx] = rx.projectFields(src, self.mesh, u)
        return data

    def projectFieldsDeriv(self, u):
        raise Exception('Use Source to project fields deriv.')
