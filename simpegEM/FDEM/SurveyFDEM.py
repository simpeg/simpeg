from SimPEG import Survey, Problem, Utils, np, sp
from simpegEM.Utils import SrcUtils
from simpegEM.Utils.EMUtils import omega, e_from_j, j_from_e, b_from_h, h_from_b
from scipy.constants import mu_0

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
    integrate = True

    def eval(self, prob):
        S_m = self.S_m(prob)
        S_e = self.S_e(prob)
        return S_m, S_e 

    def evalDeriv(self, prob, v, adjoint=False):
        return lambda v: self.S_mDeriv(prob,v,adjoint), lambda v: self.S_eDeriv(prob,v,adjoint)

    def bPrimary(self, prob):
        return None 

    def hPrimary(self, prob):
        return None

    def ePrimary(self, prob):
        return None

    def jPrimary(self, prob):
        return None

    def S_m(self, prob):
        return None

    def S_e(self, prob):
        return None

    def S_mDeriv(self, prob, v, adjoint = False):
        return None

    def S_eDeriv(self, prob, v, adjoint = False):
        return None


class SrcFDEM_RawVec_e(SrcFDEM):
    """
        RawVec electric source. It is defined by the user provided vector S_e

        :param numpy.array S_e: electric source term
        :param float freq: frequency
        :param rxList: receiver list
    """

    def __init__(self, rxList, freq, S_e):
        self._S_e = np.array(S_e,dtype=complex)
        self.freq = float(freq)
        SrcFDEM.__init__(self, rxList)

    def S_e(self, prob):
        return self._S_e


class SrcFDEM_RawVec_m(SrcFDEM):
    """
        RawVec magnetic source. It is defined by the user provided vector S_m

        :param numpy.array S_m: magnetic source term
        :param float freq: frequency
        :param rxList: receiver list
    """

    def __init__(self, rxList, freq, S_m, integrate = True):
        self._S_m = np.array(S_m,dtype=complex)
        self.freq = float(freq)
        self.integrate = integrate
        SrcFDEM.__init__(self, rxList)

    def S_m(self, prob):
        return self._S_m


class SrcFDEM_RawVec(SrcFDEM):
    """
        RawVec source. It is defined by the user provided vectors S_m, S_e

        :param numpy.array S_m: magnetic source term
        :param numpy.array S_e: electric source term
        :param float freq: frequency
        :param rxList: receiver list
    """
    def __init__(self, rxList, freq, S_m, S_e, integrate = True):
        self._S_m = np.array(S_m,dtype=complex)
        self._S_e = np.array(S_e,dtype=complex)
        self.freq = float(freq)
        self.integrate = integrate
        SrcFDEM.__init__(self, rxList)

    def S_m(self, prob):
        if prob._eqLocs is 'EF' and self.integrate is True:
            return prob.Me * self._S_m
        return self._S_m

    def S_e(self, prob):
        if prob._eqLocs is 'FE' and self.integrate is True:
            return prob.Me * self._S_e
        return self._S_e

 
class SrcFDEM_MagDipole(SrcFDEM):

    #TODO: right now, orientation doesn't actually do anything! The methods in SrcUtils should take care of that
    def __init__(self, rxList, freq, loc, orientation='Z', moment=1., mu = mu_0):
        self.freq = float(freq)
        self.loc = loc
        self.orientation = orientation
        self.moment = moment
        self.mu = mu
        self.integrate = False
        SrcFDEM.__init__(self, rxList)

    def bPrimary(self, prob):
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
            a = SrcUtils.MagneticDipoleVectorPotential(self.loc, gridY, 'y', mu=self.mu, moment=self.moment)

        else:
            srcfct = SrcUtils.MagneticDipoleVectorPotential
            ax = srcfct(self.loc, gridX, 'x', mu=self.mu, moment=self.moment)
            ay = srcfct(self.loc, gridY, 'y', mu=self.mu, moment=self.moment)
            az = srcfct(self.loc, gridZ, 'z', mu=self.mu, moment=self.moment)
            a = np.concatenate((ax, ay, az))

        return C*a

    def hPrimary(self, prob):
        b = self.bPrimary(prob)
        return h_from_b(prob,b)

    def S_m(self, prob):
        b_p = self.bPrimary(prob)
        return -1j*omega(self.freq)*b_p 

    def S_e(self, prob):
        if all(np.r_[self.mu] == np.r_[prob.curModel.mu]):
            return None
        else:
            eqLocs = prob._eqLocs

            if eqLocs is 'FE':
                mui_s = prob.curModel.mui - 1./self.mu
                MMui_s = prob.mesh.getFaceInnerProduct(mui_s)
                C = prob.mesh.edgeCurl
            elif eqLocs is 'EF':
                mu_s = prob.curModel.mu - self.mu
                MMui_s = prob.mesh.getEdgeInnerProduct(mu_s,invMat=True)
                C = prob.mesh.edgeCurl.T

            return -C.T * (MMui_s * self.bPrimary(prob))


class SrcFDEM_MagDipole_Bfield(SrcFDEM):

    #TODO: right now, orientation doesn't actually do anything! The methods in SrcUtils should take care of that
    #TODO: neither does moment
    def __init__(self, rxList, freq, loc, orientation='Z', moment=1., mu = mu_0):
        self.freq = float(freq)
        self.loc = loc
        self.orientation = orientation
        self.moment = moment
        self.mu = mu
        SrcFDEM.__init__(self, rxList)

    def bPrimary(self, prob):
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
            bx = srcfct(self.loc, gridX, 'x', mu=self.mu, moment=self.moment)
            bz = srcfct(self.loc, gridZ, 'z', mu=self.mu, moment=self.moment)
            b = np.concatenate((bx,bz))
        else:
            bx = srcfct(self.loc, gridX, 'x', mu=self.mu, moment=self.moment)
            by = srcfct(self.loc, gridY, 'y', mu=self.mu, moment=self.moment)
            bz = srcfct(self.loc, gridZ, 'z', mu=self.mu, moment=self.moment)
            b = np.concatenate((bx,by,bz))

        return b

    def hPrimary(self, prob):
        b = self.bPrimary(prob)
        return h_from_b(prob, b)

    def S_m(self, prob):
        b = self.bPrimary(prob)
        return -1j*omega(self.freq)*b

    def S_e(self, prob):
        if all(np.r_[self.mu] == np.r_[prob.curModel.mu]):
            return None
        else:
            eqLocs = prob._eqLocs

            if eqLocs is 'FE':
                mui_s = prob.curModel.mui - 1./self.mu
                MMui_s = prob.mesh.getFaceInnerProduct(mui_s)
                C = prob.mesh.edgeCurl
            elif eqLocs is 'EF':
                mu_s = prob.curModel.mu - self.mu
                MMui_s = prob.mesh.getEdgeInnerProduct(mu_s,invMat=True)
                C = prob.mesh.edgeCurl.T

            return -C.T * (MMui_s * self.bPrimary(prob))


class SrcFDEM_CircularLoop(SrcFDEM):

    #TODO: right now, orientation doesn't actually do anything! The methods in SrcUtils should take care of that
    def __init__(self, rxList, freq, loc, orientation='Z', radius = 1., mu=mu_0):
        self.freq = float(freq)
        self.orientation = orientation
        self.radius = radius
        self.mu = mu
        self.loc = loc
        self.integrate = False
        SrcFDEM.__init__(self, rxList)

    def bPrimary(self, prob):
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
            a = SrcUtils.MagneticDipoleVectorPotential(self.loc, gridY, 'y', moment=self.radius, mu=self.mu)

        else:
            srcfct = SrcUtils.MagneticDipoleVectorPotential
            ax = srcfct(self.loc, gridX, 'x', self.radius, mu=self.mu)
            ay = srcfct(self.loc, gridY, 'y', self.radius, mu=self.mu)
            az = srcfct(self.loc, gridZ, 'z', self.radius, mu=self.mu)
            a = np.concatenate((ax, ay, az))

        return C*a

    def hPrimary(self, prob):
        b = self.bPrimary(prob)
        return 1./self.mu*b

    def S_m(self, prob):
        b = self.bPrimary(prob)
        return -1j*omega(self.freq)*b

    def S_e(self, prob):
        if all(np.r_[self.mu] == np.r_[prob.curModel.mu]):
            return None
        else:
            eqLocs = prob._eqLocs

            if eqLocs is 'FE':
                mui_s = prob.curModel.mui - 1./self.mu
                MMui_s = prob.mesh.getFaceInnerProduct(mui_s)
                C = prob.mesh.edgeCurl
            elif eqLocs is 'EF':
                mu_s = prob.curModel.mu - self.mu
                MMui_s = prob.mesh.getEdgeInnerProduct(mu_s,invMat=True)
                C = prob.mesh.edgeCurl.T

            return -C.T * (MMui_s * self.bPrimary(prob))


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
