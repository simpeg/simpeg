from __future__ import division, print_function
import SimPEG
from SimPEG import np, Utils
from SimPEG.Utils import Zero, Identity
from scipy.constants import mu_0
from SimPEG.EM.Utils import *


####################################################
# Sources
####################################################


class BaseWaveform(object):

    def __init__(self, offTime=0., hasInitialFields=False, **kwargs):
        Utils.setKwargs(self, **kwargs)
        self.offTime = offTime
        self.hasInitialFields = hasInitialFields

    def _assertMatchesPair(self, pair):
        assert isinstance(self, pair), ("Waveform object must be an instance "
                                        "of a %s BaseWaveform class.".format(
                                            pair.__name__))

    def eval(self, time):
        raise NotImplementedError

    def evalDeriv(self, time):
        raise NotImplementedError  # needed for E-formulation


class StepOffWaveform(BaseWaveform):

    def __init__(self, offTime=0.):
        BaseWaveform.__init__(self, offTime, hasInitialFields=True)

    def eval(self, time):
        return 0.


class RawWaveform(BaseWaveform):

    def __init__(self, offTime=0., waveFct=None, **kwargs):
        self.waveFct = waveFct
        BaseWaveform.__init__(self, offTime, **kwargs)

    def eval(self, time):
        return self.waveFct(time)


class TriangularWaveform(BaseWaveform):

    def __init__(self, offTime=0.):
        BaseWaveform.__init__(self, offTime, hasInitialFields=True)

    def eval(self, time):
        raise NotImplementedError('TriangularWaveform has not been implemented, you should write it!')


class BaseSrc(SimPEG.Survey.BaseSrc):

    # rxPair = Rx
    integrate = True
    waveformPair = BaseWaveform

    def __init__(self, rxList, **kwargs):
        Survey.BaseSrc.__init__(self, rxList, **kwargs)

    @property
    def waveform(self):
        "A waveform instance is not None"
        return getattr(self, '_waveform', None)

    @waveform.setter
    def waveform(self, val):
        if self.waveform is None:
            val._assertMatchesPair(self.waveformPair)
            self._waveform = val
        else:
            self._waveform = self.StepOffWaveform(val)

    def __init__(self, rxList, waveform = StepOffWaveform(), **kwargs):
        self.waveform = waveform
        SimPEG.Survey.BaseSrc.__init__(self, rxList, **kwargs)

    def bInitial(self, prob):
        return Zero()

    def bInitialDeriv(self, prob, v=None, adjoint=False):
        return Zero()

    def eInitial(self, prob):
        return Zero()

    def eInitialDeriv(self, prob, v=None, adjoint=False):
        return Zero()

    def eval(self, prob, time):
        s_m = self.s_m(prob, time)
        s_e = self.s_e(prob, time)
        return s_m, s_e

    def evalDeriv(self, prob, time, v=None, adjoint=False):
        if v is not None:
            return (self.s_mDeriv(prob, time, v, adjoint),
                    self.s_eDeriv(prob, time, v, adjoint))
        else:
            return (lambda v: self.s_mDeriv(prob, time, v, adjoint),
                    lambda v: self.s_eDeriv(prob, time, v, adjoint))

    def s_m(self, prob, time):
        return Zero()

    def s_e(self, prob, time):
        return Zero()

    def s_mDeriv(self, prob, time, v=None, adjoint=False):
        return Zero()

    def s_eDeriv(self, prob, time, v=None, adjoint=False):
        return Zero()


class MagDipole(BaseSrc):

    waveform = None
    loc = None
    orientation = 'Z'
    moment = 1.
    mu = mu_0

    def __init__(self, rxList, **kwargs):
        assert(self.orientation in ['X', 'Y', 'Z']), (
            "Orientation (right now) doesn't actually do anything! The methods"
            " in SrcUtils should take care of this..."
            )
        self.integrate = False
        BaseSrc.__init__(self, rxList, **kwargs)

    def _bfromVectorPotential(self, prob):
        if prob._eqLocs is 'FE':
            gridX = prob.mesh.gridEx
            gridY = prob.mesh.gridEy
            gridZ = prob.mesh.gridEz
            C = prob.mesh.edgeCurl

        elif prob._eqLocs is 'EF':
            gridX = prob.mesh.gridFx
            gridY = prob.mesh.gridFy
            gridZ = prob.mesh.gridFz
            C = prob.mesh.edgeCurl.T

        if prob.mesh._meshType is 'CYL':
            if not prob.mesh.isSymmetric:
                raise NotImplementedError('Non-symmetric cyl mesh '
                                          'not implemented yet!')
            a = MagneticDipoleVectorPotential(self.loc, gridY, 'y', mu=self.mu,
                                              moment=self.moment)

        else:
            srcfct = MagneticDipoleVectorPotential
            ax = srcfct(self.loc, gridX, 'x', mu=self.mu, moment=self.moment)
            ay = srcfct(self.loc, gridY, 'y', mu=self.mu, moment=self.moment)
            az = srcfct(self.loc, gridZ, 'z', mu=self.mu, moment=self.moment)
            a = np.concatenate((ax, ay, az))

        return C*a

    def bInitial(self, prob):

        if self.waveform.hasInitialFields is False:
            return Zero()

        return self._bfromVectorPotential(prob)

    def eInitial(self, prob):
        # when solving for e, it is easier to work with an initial source than
        # initial fields
        # if self.waveform.hasInitialFields is False or prob._fieldType is 'e':
        return Zero()

        # b = self.bInitial(prob)
        # MeSigmaI = prob.MeSigmaI
        # MfMui = prob.MfMui
        # C = prob.mesh.edgeCurl

        # return MeSigmaI * (C.T * (MfMui * b))

    def eInitialDeriv(self, prob, v=None, adjoint=False):

        return Zero()

        # if self.waveform.hasInitialFields is False:
        #     return Zero()

        # b = self.bInitial(prob)
        # MeSigmaIDeriv = prob.MeSigmaIDeriv
        # MfMui = prob.MfMui
        # C = prob.mesh.edgeCurl
        # s_e = self.s_e(prob, prob.t0)

        # # s_e doesn't depend on the model

        # if adjoint:
        #     return MeSigmaIDeriv( -s_e + C.T * ( MfMui * b ) ).T * v

        # return MeSigmaIDeriv( -s_e + C.T * ( MfMui * b ) ) * v

    def s_m(self, prob, time):
        if self.waveform.hasInitialFields is False:
            # raise NotImplementedError
            return Zero()
        return Zero()

    def s_e(self, prob, time):
        b = self._bfromVectorPotential(prob)
        MfMui = prob.MfMui
        C = prob.mesh.edgeCurl

        # print 'time ', time

        if self.waveform.hasInitialFields is True and time < prob.timeSteps[1]:
            # if time > 0.0:
            #     return Zero()
            if prob._fieldType == 'b':
                return Zero()
            elif prob._fieldType == 'e':
                # Compute s_e from vector potential
                return C.T * (MfMui * b)
        else:
            # b = self._bfromVectorPotential(prob)
            return C.T * (MfMui * b) * self.waveform.eval(time)
        # return Zero()


class CircularLoop(MagDipole):

    waveform = None
    loc = None
    orientation = 'Z'
    radius = None
    mu = mu_0

    def __init__(self, rxList, **kwargs):
        assert(self.orientation in ['X', 'Y', 'Z']), (
            "Orientation (right now) doesn't actually do anything! The methods"
            " in SrcUtils should take care of this..."
            )
        self.integrate = False
        BaseSrc.__init__(self, rxList, **kwargs)

    def _bfromVectorPotential(self, prob):
        if prob._eqLocs is 'FE':
            gridX = prob.mesh.gridEx
            gridY = prob.mesh.gridEy
            gridZ = prob.mesh.gridEz
            C = prob.mesh.edgeCurl

        elif prob._eqLocs is 'EF':
            gridX = prob.mesh.gridFx
            gridY = prob.mesh.gridFy
            gridZ = prob.mesh.gridFz
            C = prob.mesh.edgeCurl.T

        if prob.mesh._meshType is 'CYL':
            if not prob.mesh.isSymmetric:
                raise NotImplementedError('Non-symmetric cyl mesh not '
                                          'implemented yet!')
            a = MagneticLoopVectorPotential(self.loc, gridY, 'y',
                                            radius=self.radius, mu=self.mu)

        else:
            srcfct = MagneticLoopVectorPotential
            ax = srcfct(self.loc, gridX, 'x', mu=self.mu, radius=self.radius)
            ay = srcfct(self.loc, gridY, 'y', mu=self.mu, radius=self.radius)
            az = srcfct(self.loc, gridZ, 'z', mu=self.mu, radius=self.radius)
            a = np.concatenate((ax, ay, az))

        return C*a


class LineCurrent(BaseSrc):
    """
    RawVec electric source. It is defined by the user provided vector s_e

    :param list rxList: receiver list
    :param bool integrate: Integrate the source term (multiply by Me) [False]
    """
    waveform = None
    loc = None
    mu = mu_0

    def __init__(self, rxList, **kwargs):
        self.integrate = False
        BaseSrc.__init__(self, rxList, **kwargs)

    def getAdc(self, prob):
        MeSigma = self.prob.MeSigma
        Grad = self.prob.mesh.nodalGrad
        Adc = Grad.T * MeSigma * Grad
        # Handling Null space of A
        Adc[0, 0] = Adc[0, 0] + 1.
        return Adc

    def _Mejs(self, prob):
        if getattr(self, '_Mejs', None) is None:
            x0 = prob.mesh.x0
            hx = prob.mesh.hx
            hy = prob.mesh.hy
            hz = prob.mesh.hz
            px = loc[:, 0]
            py = loc[:, 1]
            pz = loc[:, 2]
            self._Mejs = getSourceTermLineCurrentPolygon(x0, hx, hy, hz,
                                                         px, py, pz)
        return self._Mejs

    def getRHSdc(self, prob):
        Grad = self.prob.nodalGrad
        return Grad.T*self.Mejs

    def _getInitialFields(self, prob):
        # TODO: Be careful about what to store, and we delete stuff
        # Because it is expensive to compute ...
        # e.g. we only need to update initial field when "m" is changed

        # Solve DCR problem
        Adc = getAdc(prob)
        # TODO: Ainvdc may need to be stored in problem class
        # "We have multiple src"
        Ainvdc = prob.Solver(Adc, **prob.solverOpts)
        rhsdc = self.getRHSdc(prob)
        phidc = Ainvdc*rhsdc
        Grad = self.prob.nodalGrad
        edc = -Grad*phidc
        if prob._fieldType == 'e':
            return edc
        # Solve MMR problem
        elif prob._fieldType == 'b':
            C = prob.edgeCurl
            MfMui = prob.MfMui
            MeSigma = self.prob.MeSigma
            # Second term on rhs handles null space
            Ammr = C.T*MfMui*C + 1./mu_0 * prob.Me * Grad * Grad.T
            rhsmmr = self.MeSigma*edc + self.Mejs
            # TODO: Ainvmmr may need to be stored in problem class
            # "We have multiple src"
            Ainvmmr = prob.Solver(Ammr, **prob.solverOpts)
            bmmr = Ainvmmr*rhsmmr
            return bmmr
        else:
            raise NotImplementedError("We only have EB formulation!")

    def bInitial(self, prob):
        if self.waveform.hasInitialFields is False:
            return Zero()
        return self._getInitialFields(prob)

    def eInitial(self, prob):
        # when solving for e, it is easier to work with an initial source than
        # initial fields
        # if self.waveform.hasInitialFields is False or prob._fieldType is 'e':
        if self.waveform.hasInitialFields is False:
            return Zero()
        return self._getInitialFields(prob)

    def eInitialDeriv(self, prob, v=None, adjoint=False):
        if self.waveform.hasInitialFields is False:
            return Zero()
        pass
        # b = self.bInitial(prob)
        # MeSigmaIDeriv = prob.MeSigmaIDeriv
        # MfMui = prob.MfMui
        # C = prob.mesh.edgeCurl
        # s_e = self.s_e(prob, prob.t0)

        # # s_e doesn't depend on the model

        # if adjoint:
        #     return MeSigmaIDeriv( -s_e + C.T * ( MfMui * b ) ).T * v

        # return MeSigmaIDeriv( -s_e + C.T * ( MfMui * b ) ) * v

    def s_m(self, prob, time):
        return Zero()

    def s_e(self, prob, time):
        return self.Mejs * self.waveform.eval(time)
