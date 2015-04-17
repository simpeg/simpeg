from SimPEG import Survey, Problem, Utils, np, sp
from simpegEM import Sources
from simpegEM.Utils.EMUtils import omega


####################################################
# Receivers 
####################################################

class RxFDEM(Survey.BaseRx):

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
        Survey.BaseRx.__init__(self, locs, rxType)

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
# Sources
####################################################

# class SrcFDEM(Survey.BaseSrc):
#     freq = None
#     rxPair = RxFDEM
#     knownSrcTypes = {}


class SrcFDEM(Survey.BaseSrc):
    #TODO: Break these out into Classes of Sources.

    freq = None #: Frequency (float)

    rxPair = RxFDEM

    knownSrcTypes = ['VMD', 'VMD_B', 'CircularLoop', 'Simple']

    radius = None

    def __init__(self, loc, srcType, freq, rxList):
        self.freq = float(freq)
        Survey.BaseSrc.__init__(self, loc, srcType, rxList)

    def getSource(self, prob):

        src = self
        freq = src.freq
        solType = prob._fieldType # Hack, should just ask whether j_m, j_g are defined on edges or faces

        if solType == 'e' or solType == 'b':
            gridEJx = prob.mesh.gridEx
            gridEJy = prob.mesh.gridEy
            gridEJz = prob.mesh.gridEz
            nEJ = prob.mesh.nE

            gridBHx = prob.mesh.gridFx
            gridBHy = prob.mesh.gridFy
            gridBHz = prob.mesh.gridFz
            nBH = prob.mesh.nF


            C = prob.mesh.edgeCurl
            mui = prob.MfMui

        elif solType == 'h' or solType == 'j':
            gridEJx = prob.mesh.gridFx
            gridEJy = prob.mesh.gridFy
            gridEJz = prob.mesh.gridFz
            nEJ = prob.mesh.nF

            gridBHx = prob.mesh.gridEx
            gridBHy = prob.mesh.gridEy
            gridBHz = prob.mesh.gridEz
            nBH = prob.mesh.nE

            C = prob.mesh.edgeCurl.T
            mui = prob.MeMuI

        else:
            NotImplementedError('Only E or F sources')


        if prob.mesh._meshType is 'CYL':
            if not prob.mesh.isSymmetric:
                raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')

            if src.srcType == 'VMD':
                SRC = Sources.MagneticDipoleVectorPotential(src.loc, gridEJy, 'y')
            elif src.srcType == 'CircularLoop':
                SRC = Sources.MagneticLoopVectorPotential(src.loc, gridEJy, 'y', src.radius)
            else:
                raise NotImplementedError('Only VMD and CircularLoop')

        elif prob.mesh._meshType is 'TENSOR':

            if src.srcType == 'VMD':
                srcfct = Sources.MagneticDipoleVectorPotential
                SRCx = srcfct(src.loc, gridEJx, 'x')
                SRCy = srcfct(src.loc, gridEJy, 'y')
                SRCz = srcfct(src.loc, gridEJz, 'z')

            elif src.srcType == 'VMD_B':
                srcfct = Sources.MagneticDipoleFields
                SRCx = srcfct(src.loc, gridBHx, 'x')
                SRCy = srcfct(src.loc, gridBHy, 'y')
                SRCz = srcfct(src.loc, gridBHz, 'z')

            elif src.srcType == 'CircularLoop':
                srcfct = Sources.MagneticLoopVectorPotential
                SRCx = srcfct(src.loc, gridEJx, 'x', src.radius)
                SRCy = srcfct(src.loc, gridEJy, 'y', src.radius)
                SRCz = srcfct(src.loc, gridEJz, 'z', src.radius)
            else:

                raise NotImplemented('%s srcType is not implemented' % src.srcType)
            SRC = np.concatenate((SRCx, SRCy, SRCz))

        else:
            raise Exception('Unknown mesh for VMD')

        # b-forumlation
        if src.srcType == 'VMD_B':
            b_0 = SRC
        else:
            a = SRC
            b_0 = C*a

        return -1j*omega(freq)*b_0, None

class SimpleSrcFDEM_e(SrcFDEM):

    def __init__(self, vec, freq, rxList):
        self.vec = vec
        self.freq = float(freq)
        SrcFDEM.__init__(self, None, 'Simple', freq, rxList)

    def getSource(self, prob):
        return None, self.vec


class SimpleSrcFDEM_m(SrcFDEM):

    def __init__(self, vec, freq, rxList):
        self.vec = vec
        self.freq = float(freq)
        SrcFDEM.__init__(self, None, 'Simple', freq, rxList)

    def getSource(self, prob):
        return self.vec, None


class SurveyFDEM(Survey.BaseSurvey):
    """
        docstring for SurveyFDEM
    """

    srcPair = SrcFDEM

    def __init__(self, srcList, **kwargs):
        # Sort these by frequency
        self.srcList = srcList
        Survey.BaseSurvey.__init__(self, **kwargs)

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
                self._nSrcByFreq[freq] = len(self.getSources(freq))
        return self._nSrcByFreq

    def getSources(self, freq):
        """Returns the sources associated with a specific frequency."""
        assert freq in self._freqDict, "The requested frequency is not in this survey."
        return self._freqDict[freq]

    def projectFields(self, u):
        data = Survey.Data(self)
        for src in self.srcList:
            for rx in src.rxList:
                data[src, rx] = rx.projectFields(src, self.mesh, u)
        return data

    def projectFieldsDeriv(self, u):
        raise Exception('Use Sources to project fields deriv.')
