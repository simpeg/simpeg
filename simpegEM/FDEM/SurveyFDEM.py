from SimPEG import Survey, Problem, Utils, np, sp
from simpegEM.Utils import SrcUtils
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

class SrcFDEM(Survey.BaseSrc):
    freq = None
    rxPair = RxFDEM
    knownSrcTypes = ['Simple', 'MagDipole'] #TODO: Do we want to just classify them by Magnetic, Electric, Both? 


class SrcFDEM_Simple_e(SrcFDEM):
    """
        Simple electric source. It is defined by the user provided vector S_e

        :param numpy.array S_e: electric source term
        :param float freq: frequency
        :param rxList: receiver list
    """

    def __init__(self, S_e, freq, rxList):
        self.S_e = S_e
        self.freq = float(freq)
        SrcFDEM.__init__(self, None, 'Simple', rxList)

    def getSource(self, prob):
        return None, self.S_e

    def getSourceDeriv(self, prob, v, adjoint = False):
        return None, None


class SrcFDEM_Simple_m(SrcFDEM):
    """
        Simple magnetic source. It is defined by the user provided vector S_m

        :param numpy.array S_m: magnetic source term
        :param float freq: frequency
        :param rxList: receiver list
    """

    def __init__(self, S_m, freq, rxList):
        self.S_m = S_m
        self.freq = float(freq)
        SrcFDEM.__init__(self, None, 'Simple', rxList)

    def getSource(self, prob):
        return self.S_m, None

    def getSourceDeriv(self, prob, v, adjoint = False):
        return None, None


class SrcFDEM_Simple(SrcFDEM):
    """
        Simple source. It is defined by the user provided vectors S_m, S_e

        :param numpy.array S_m: magnetic source term
        :param numpy.array S_e: electric source term
        :param float freq: frequency
        :param rxList: receiver list
    """
    def __init__(self, S_m, S_e, freq, rxList):
        self.S_m = S_m
        self.S_e = S_e
        SrcFDEM.__init__(self, None, 'Simple', rxList)

    def getSource(self, prob):
        return self.S_m, self.S_e

    def getSourceDeriv(self, prob, v, adjoint=None):
        return None, None


class SrcFDEM_MagDipole(SrcFDEM):

    #TODO: right now, orientation doesn't actually do anything! The methods in SrcUtils should take care of that
    def __init__(self, loc, freq, rxList, orientation='Z'):
        self.freq = float(freq)
        self.orientation = orientation
        SrcFDEM.__init__(self, loc, 'MagDipole', rxList)

    def getSource(self, prob):
        eqLocs = prob._eqLocs

        if eqLocs is 'FE':
            gridX = prob.mesh.gridEx
            gridY = prob.mesh.gridEy
            gridZ = prob.mesh.gridEz
            C = prob.mesh.edgeCurl

        elif eqLocs is 'EF':
            gridX = prob.mesh.gridFx
            gridY = prob.mesh.gridFy
            gridZ = prob.mesh.gridFz
            C = prob.mesh.edgeCurl.T


        if prob.mesh._meshType is 'CYL':
            if not prob.mesh.isSymmetric:
                # TODO ?
                raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')
            a = SrcUtils.MagneticDipoleVectorPotential(src.loc, gridY, 'y')

        else:
            srcfct = SrcUtils.MagneticDipoleVectorPotential
            ax = srcfct(self.loc, gridX, 'x')
            ay = srcfct(self.loc, gridY, 'y')
            az = srcfct(self.loc, gridZ, 'z')
            a = np.concatenate((ax, ay, az))

        S_m = -1j*omega(self.freq)*C*a

        return S_m, None


    def getSourceDeriv(self, prob, v, adjoint=None):
        return None, None


class MagDipole_Bfield(SrcFDEM):

    #TODO: right now, orientation doesn't actually do anything! The methods in SrcUtils should take care of that
    def __init__(self, loc, freq, rxList, orientation='Z'):
        self.freq = float(freq)
        self.orientation = orientation
        SrcFDEM.__init__(self, loc, 'MagDipole', rxList)

    def getSource(self, prob):
        eqLocs = prob._eqLocs

        if eqLocs is 'FE':
            gridX = prob.mesh.gridFx
            gridY = prob.mesh.gridFy
            gridZ = prob.mesh.gridFz
            C = prob.mesh.edgeCurl

        elif eqLocs is 'EF':
            gridX = prob.mesh.gridEx
            gridY = prob.mesh.gridEy
            gridZ = prob.mesh.gridEz
            C = prob.mesh.edgeCurl.T

        srcfct = SrcUtils.MagneticDipoleFields
        if prob.mesh._meshType is 'CYL':
            if not prob.mesh.isSymmetric:
                # TODO ?
                raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')
            bx = srcfct(self.loc, gridX, 'x')
            bz = srcfct(self.loc, gridZ, 'z')
            b = np.concatenate((bx,bz))
        else:
            bx = srcfct(self.loc, gridX, 'x')
            by = srcfct(self.loc, gridY, 'y')
            bz = srcfct(self.loc, gridZ, 'z')
            b = np.concatenate((bx,by,bz))

        return -1j*omega(self.freq)*b, None

    def getSourceDeriv(self, prob, v, adjoint=None):
        return None, None


class SrcFDEM_CircularLoop(SrcFDEM):

    #TODO: right now, orientation doesn't actually do anything! The methods in SrcUtils should take care of that
    def __init__(self, loc, freq, rxList, orientation='Z', radius = 1.):
        self.freq = float(freq)
        self.orientation = orientation
        self.radius = radius
        SrcFDEM.__init__(self, loc, 'MagDipole', rxList)

    def getSource(self, prob):
        eqLocs = prob._eqLocs

        if eqLocs is 'FE':
            gridX = prob.mesh.gridEx
            gridY = prob.mesh.gridEy
            gridZ = prob.mesh.gridEz
            C = prob.mesh.edgeCurl

        elif eqLocs is 'EF':
            gridX = prob.mesh.gridFx
            gridY = prob.mesh.gridFy
            gridZ = prob.mesh.gridFz
            C = prob.mesh.edgeCurl.T

        if prob.mesh._meshType is 'CYL':
            if not prob.mesh.isSymmetric:
                # TODO ?
                raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')
            a = SrcUtils.MagneticDipoleVectorPotential(src.loc, gridY, 'y', self.radius)

        else:
            srcfct = SrcUtils.MagneticDipoleVectorPotential
            ax = srcfct(self.loc, gridX, 'x', self.radius)
            ay = srcfct(self.loc, gridY, 'y', self.radius)
            az = srcfct(self.loc, gridZ, 'z', self.radius)
            a = np.concatenate((ax, ay, az))

        return -1j*omega(self.freq)*C*a


####################################################
# Survey
####################################################

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
