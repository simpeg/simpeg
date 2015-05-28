from SimPEG import Survey, Problem, Utils, np, sp
from simpegEM.Utils import SrcUtils
from simpegEM.Utils.EMUtils import omega, e_from_j, j_from_e, b_from_h, h_from_b


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

    def eval(self, prob):
        S_m = self._getS_m(prob)
        S_e = self._getS_e(prob)

        if S_m is not None and S_m.ndim == 1: S_m = Utils.mkvc(S_m,2)
        if S_e is not None and S_e.ndim == 1: S_e = Utils.mkvc(S_e,2)

        return S_m, S_e 

    def evalDeriv(self, prob, v, adjoint=None):
        return self._getS_mDeriv(prob,v,adjoint), self._getS_eDeriv(prob,v,adjoint)

    def b_p(self,prob):
        b_p = self._getb_p(prob)
        if b_p is not None and b_p.ndim == 1: b_p = Utils.mkvc(b_p,2)
        return b_p 

    def h_p(self,prob):
        h_p = self._geth_p(prob)
        if h_p is not None and h_p.ndim == 1: h_p = Utils.mkvc(h_p,2)
        return h_p

    def e_p(self,prob):
        e_p = self._gete_p(prob)
        if e_p is not None and e_p.ndim == 1: e_p = Utils.mkvc(e_p,2)
        return e_p

    def j_p(self,prob):
        j_p = self._getj_p(prob)
        if j_p is not None and j_p.ndim == 1: j_p = Utils.mkvc(j_p,2)
        return j_p

    def _getb_p(self,prob):
        return None

    def _geth_p(self,prob):
        return None

    def _gete_p(self,prob):
        return None

    def _getj_p(self,prob):
        return None

    def _getS_m(self,prob):
        return None

    def _getS_e(self,prob):
        return None

    def _getS_mDeriv(self, prob, v, adjoint = False):
        return None

    def _getS_eDeriv(self, prob, v, adjoint = False):
        return None


class SrcFDEM_RawVec_e(SrcFDEM):
    """
        RawVec electric source. It is defined by the user provided vector S_e

        :param numpy.array S_e: electric source term
        :param float freq: frequency
        :param rxList: receiver list
    """

    def __init__(self, rxList, freq, S_e):
        self.S_e = np.array(S_e,dtype=float)
        self.freq = float(freq)
        SrcFDEM.__init__(self, rxList)

    def _getS_e(self, prob):
        return self.S_e


class SrcFDEM_RawVec_m(SrcFDEM):
    """
        RawVec magnetic source. It is defined by the user provided vector S_m

        :param numpy.array S_m: magnetic source term
        :param float freq: frequency
        :param rxList: receiver list
    """

    def __init__(self, rxList, freq, S_m):
        self.S_m = np.array(S_m,dtype=float)
        self.freq = float(freq)
        SrcFDEM.__init__(self, rxList)

    def _getS_m(self, prob):
        return self.S_m


class SrcFDEM_RawVec(SrcFDEM):
    """
        RawVec source. It is defined by the user provided vectors S_m, S_e

        :param numpy.array S_m: magnetic source term
        :param numpy.array S_e: electric source term
        :param float freq: frequency
        :param rxList: receiver list
    """
    def __init__(self, rxList, freq, S_m, S_e):
        self.S_m = np.array(S_m,dtype=float)
        self.S_e = np.array(S_e,dtype=float)
        self.freq = float(freq)
        SrcFDEM.__init__(self, rxList)

    def _getS_m(self,prob):
        return self.S_m

    def _getS_e(self,prob):
        return self.S_e

 
class SrcFDEM_MagDipole(SrcFDEM):

    #TODO: right now, orientation doesn't actually do anything! The methods in SrcUtils should take care of that
    def __init__(self, rxList, freq, loc, orientation='Z', moment=1.):
        self.freq = float(freq)
        self.loc = loc
        self.orientation = orientation
        self.moment = moment
        SrcFDEM.__init__(self, rxList)

    def _getb_p(self,prob):
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
            a = SrcUtils.MagneticDipoleVectorPotential(self.loc, gridY, 'y')

        else:
            srcfct = SrcUtils.MagneticDipoleVectorPotential
            ax = srcfct(self.loc, gridX, 'x')
            ay = srcfct(self.loc, gridY, 'y')
            az = srcfct(self.loc, gridZ, 'z')
            a = np.concatenate((ax, ay, az))

        return C*a

    def _geth_p(self,prob):
        b = self._getb_p(prob)
        return h_from_b(prob,b)

    def _getS_m(self,prob):
        b_p = self._getb_p(prob)
        return -1j*omega(self.freq)*b_p 



class SrcFDEM_MagDipole_Bfield(SrcFDEM):

    #TODO: right now, orientation doesn't actually do anything! The methods in SrcUtils should take care of that
    #TODO: neither does moment
    def __init__(self, rxList, freq, loc, orientation='Z', moment=1.):
        self.freq = float(freq)
        self.loc = loc
        self.orientation = orientation
        self.moment = moment
        SrcFDEM.__init__(self, rxList)

    def _getb_p(self,prob):
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

        return b

    def _geth_p(self,prob):
        b = self._getb_p(prob)
        return h_from_b(prob, b)

    def _getS_m(self,prob):
        b = self._getb_p(prob)
        return -1j*omega(self.freq)*b


class SrcFDEM_CircularLoop(SrcFDEM):

    #TODO: right now, orientation doesn't actually do anything! The methods in SrcUtils should take care of that
    def __init__(self, rxList, freq, loc, orientation='Z', radius = 1.):
        self.freq = float(freq)
        self.orientation = orientation
        self.radius = radius
        SrcFDEM.__init__(self, rxList)

    def _getb_p(self,prob):
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

        return C*a

    def _geth_p(self,prob):
        b = self._getb_p(prob)
        return h_from_b

    def _getS_m(self, prob):
        b = self._getb_p(prob)
        return -1j*omega(self.freq)*b


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
                self._nSrcByFreq[freq] = len(self.getSrcByFreq(freq))
        return self._nSrcByFreq

    def getSrcByFreq(self, freq):
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
