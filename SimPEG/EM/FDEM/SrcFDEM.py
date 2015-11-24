from SimPEG import Survey, Problem, Utils, np, sp
from scipy.constants import mu_0
from SimPEG.EM.Utils import *
from SimPEG.Utils import Zero
# from SurveyFDEM import Rx
 

class BaseSrc(Survey.BaseSrc):
    freq = None
    # rxPair = Rx
    integrate = True

    def eval(self, prob):
        S_m = self.S_m(prob)
        S_e = self.S_e(prob)
        return S_m, S_e

    def evalDeriv(self, prob, v, adjoint=False):
        return lambda v: self.S_mDeriv(prob,v,adjoint), lambda v: self.S_eDeriv(prob,v,adjoint)

    def bPrimary(self, prob):
        return Zero()

    def hPrimary(self, prob):
        return Zero()

    def ePrimary(self, prob):
        return Zero()

    def jPrimary(self, prob):
        return Zero()

    def S_m(self, prob):
        return Zero()

    def S_e(self, prob):
        return Zero()

    def S_mDeriv(self, prob, v, adjoint = False):
        return Zero()

    def S_eDeriv(self, prob, v, adjoint = False):
        return Zero()


class RawVec_e(BaseSrc):
    """
        RawVec electric source. It is defined by the user provided vector S_e

        :param numpy.array S_e: electric source term
        :param float freq: frequency
        :param rxList: receiver list
    """

    def __init__(self, rxList, freq, S_e): #, ePrimary=None, bPrimary=None, hPrimary=None, jPrimary=None):
        self._S_e = np.array(S_e,dtype=complex)
        self.freq = float(freq)
        BaseSrc.__init__(self, rxList)

    def S_e(self, prob):
        return self._S_e


class RawVec_m(BaseSrc):
    """
        RawVec magnetic source. It is defined by the user provided vector S_m

        :param numpy.array S_m: magnetic source term
        :param float freq: frequency
        :param rxList: receiver list
    """

    def __init__(self, rxList, freq, S_m, integrate = True):  #ePrimary=Zero(), bPrimary=Zero(), hPrimary=Zero(), jPrimary=Zero()):
        self._S_m = np.array(S_m,dtype=complex)
        self.freq = float(freq)
        self.integrate = integrate

        BaseSrc.__init__(self, rxList)

    def S_m(self, prob):
        return self._S_m


class RawVec(BaseSrc):
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
        BaseSrc.__init__(self, rxList)

    def S_m(self, prob):
        if prob._eqLocs is 'EF' and self.integrate is True:
            return prob.Me * self._S_m
        return self._S_m

    def S_e(self, prob):
        if prob._eqLocs is 'FE' and self.integrate is True:
            return prob.Me * self._S_e
        return self._S_e


class MagDipole(BaseSrc):

    #TODO: right now, orientation doesn't actually do anything! The methods in SrcUtils should take care of that
    def __init__(self, rxList, freq, loc, orientation='Z', moment=1., mu = mu_0):
        self.freq = float(freq)
        self.loc = loc
        self.orientation = orientation
        self.moment = moment
        self.mu = mu
        self.integrate = False
        BaseSrc.__init__(self, rxList)

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
            a = MagneticDipoleVectorPotential(self.loc, gridY, 'y', mu=self.mu, moment=self.moment)

        else:
            srcfct = MagneticDipoleVectorPotential
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
            return Zero()
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


class MagDipole_Bfield(BaseSrc):

    #TODO: right now, orientation doesn't actually do anything! The methods in SrcUtils should take care of that
    #TODO: neither does moment
    def __init__(self, rxList, freq, loc, orientation='Z', moment=1., mu = mu_0):
        self.freq = float(freq)
        self.loc = loc
        self.orientation = orientation
        self.moment = moment
        self.mu = mu
        BaseSrc.__init__(self, rxList)

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

        srcfct = MagneticDipoleFields
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
            return Zero()
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


class CircularLoop(BaseSrc):

    #TODO: right now, orientation doesn't actually do anything! The methods in SrcUtils should take care of that
    def __init__(self, rxList, freq, loc, orientation='Z', radius = 1., mu=mu_0):
        self.freq = float(freq)
        self.orientation = orientation
        self.radius = radius
        self.mu = mu
        self.loc = loc
        self.integrate = False
        BaseSrc.__init__(self, rxList)

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
            a = MagneticDipoleVectorPotential(self.loc, gridY, 'y', moment=self.radius, mu=self.mu)

        else:
            srcfct = MagneticDipoleVectorPotential
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
            return Zero()
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


