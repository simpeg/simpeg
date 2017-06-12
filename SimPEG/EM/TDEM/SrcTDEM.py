from __future__ import division, print_function

import numpy as np
from scipy.constants import mu_0
import properties
import warnings

from SimPEG import Utils
from SimPEG.Utils import Zero, Identity
from SimPEG.EM.Utils import *
from ..Base import BaseEMSrc


###############################################################################
#                                                                             #
#                           Source Waveforms                                  #
#                                                                             #
###############################################################################


class BaseWaveform(properties.HasProperties):

    hasInitialFields = properties.Bool(
        "Does the waveform have initial fields?", default=False
    )

    offTime = properties.Float(
        "off-time of the source", default=0.
    )

    eps = properties.Float(
        "window of time within which the waveform is considered on",
        default=1e-9
    )

    def __init__(self, **kwargs):
        Utils.setKwargs(self, **kwargs)

    def _assertMatchesPair(self, pair):
        assert isinstance(self, pair), (
            "Waveform object must be an instance of a %s "
            "BaseWaveform class.".format(pair.__name__)
        )

    def eval(self, time):
        raise NotImplementedError

    def evalDeriv(self, time):
        raise NotImplementedError  # needed for E-formulation


class StepOffWaveform(BaseWaveform):

    def __init__(self, offTime=0.):
        BaseWaveform.__init__(self, offTime=offTime, hasInitialFields=True)

    def eval(self, time):
        if abs(time-0.) < self.eps:
            return 1.
        else:
            return 0.


class RampOffWaveform(BaseWaveform):

    def __init__(self, offTime=0.):
        BaseWaveform.__init__(self, offTime=offTime, hasInitialFields=True)

    def eval(self, time):
        if abs(time-0.) < self.eps:
            return 1.
        elif time < self.offTime:
            return -1. / self.offTime * (time - self.offTime)
        else:
            return 0.


class RawWaveform(BaseWaveform):

    def __init__(self, offTime=0., waveFct=None, **kwargs):
        self.waveFct = waveFct
        BaseWaveform.__init__(self, offTime=offTime, **kwargs)

    def eval(self, time):
        return self.waveFct(time)


class TriangularWaveform(BaseWaveform):

    def __init__(self, offTime=0.):
        BaseWaveform.__init__(self, offTime, hasInitialFields=True)

    def eval(self, time):
        raise NotImplementedError(
            'TriangularWaveform has not been implemented, you should write it!'
        )


class VTEMWaveform(BaseWaveform):

    offTime = properties.Float(
        "off-time of the source", default=4.2e-3
    )

    peakTime = properties.Float(
        "Time at which the VTEM waveform is at its peak", default=2.73e-3
    )

    a = properties.Float(
         "parameter controlling how quickly the waveform ramps on", default=3.
    )

    def __init__(self, **kwargs):
        BaseWaveform.__init__(self, hasInitialFields=False, **kwargs)

    def eval(self, time):
        if time <= self.peakTime:
            return (
                (1. - np.exp(-self.a*time/self.peakTime))/(1.-np.exp(-self.a))
                )
        elif (time < self.offTime) and (time > self.peakTime):
            return -1. / (self.offTime-self.peakTime) * (time - self.offTime)
        else:
            return 0.


###############################################################################
#                                                                             #
#                                    Sources                                  #
#                                                                             #
###############################################################################

class BaseTDEMSrc(BaseEMSrc):

    # rxPair = Rx

    waveformPair = BaseWaveform  #: type of waveform to pair with
    waveform = None  #: source waveform
    srcType = None

    def __init__(self, rxList, **kwargs):
        super(BaseTDEMSrc, self).__init__(rxList, **kwargs)

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

    def __init__(self, rxList, waveform=StepOffWaveform(), **kwargs):
        self.waveform = waveform
        BaseEMSrc.__init__(self, rxList, **kwargs)

    def bInitial(self, prob):
        return Zero()

    def bInitialDeriv(self, prob, v=None, adjoint=False, f=None):
        return Zero()

    def eInitial(self, prob):
        return Zero()

    def eInitialDeriv(self, prob, v=None, adjoint=False, f=None):
        return Zero()

    def hInitial(self, prob):
        return Zero()

    def hInitialDeriv(self, prob, v=None, adjoint=False, f=None):
        return Zero()

    def jInitial(self, prob):
        return Zero()

    def jInitialDeriv(self, prob, v=None, adjoint=False, f=None):
        return Zero()

    def eval(self, prob, time):
        s_m = self.s_m(prob, time)
        s_e = self.s_e(prob, time)
        return s_m, s_e

    def evalDeriv(self, prob, time, v=None, adjoint=False):
        if v is not None:
            return (
                self.s_mDeriv(prob, time, v, adjoint),
                self.s_eDeriv(prob, time, v, adjoint)
            )
        else:
            return (
                lambda v: self.s_mDeriv(prob, time, v, adjoint),
                lambda v: self.s_eDeriv(prob, time, v, adjoint)
            )

    def s_m(self, prob, time):
        return Zero()

    def s_e(self, prob, time):
        return Zero()

    def s_mDeriv(self, prob, time, v=None, adjoint=False):
        return Zero()

    def s_eDeriv(self, prob, time, v=None, adjoint=False):
        return Zero()


class MagDipole(BaseTDEMSrc):

    moment = properties.Float(
        "dipole moment of the transmitter", default=1., min=0.
    )
    mu = properties.Float(
        "permeability of the background", default=mu_0, min=0.
        )
    orientation = properties.Vector3(
        "orientation of the source", default='Z', length=1., required=True
    )
    srcType = "Inductive"

    def __init__(self, rxList, **kwargs):
        # assert(self.orientation in ['X', 'Y', 'Z']), (
        #     "Orientation (right now) doesn't actually do anything! The methods"
        #     " in SrcUtils should take care of this..."
        #     )
        # self.integrate = False
        BaseTDEMSrc.__init__(self, rxList, **kwargs)

    @properties.validator('orientation')
    def _warn_non_axis_aligned_sources(self, change):
        value = change['value']
        axaligned = [
            True for vec in [np.r_[1.,0.,0.], np.r_[0.,1.,0.], np.r_[0.,0.,1.]]
            if np.all(value == vec)
        ]
        if len(axaligned) != 1:
            warnings.warn(
                'non-axes aligned orientations {} are not rigorously'
                ' tested'.format(value)
            )

    def _srcFct(self, obsLoc, component):
        return MagneticDipoleVectorPotential(
            self.loc, obsLoc, component, mu=self.mu, moment=self.moment
        )

    def _aSrc(self, prob):
        if prob._formulation == 'EB':
            gridX = prob.mesh.gridEx
            gridY = prob.mesh.gridEy
            gridZ = prob.mesh.gridEz

        elif prob._formulation == 'HJ':
            gridX = prob.mesh.gridFx
            gridY = prob.mesh.gridFy
            gridZ = prob.mesh.gridFz

        if prob.mesh._meshType is 'CYL':
            if not prob.mesh.isSymmetric:
                raise NotImplementedError(
                    'Non-symmetric cyl mesh not implemented yet!'
                )
            a = self._srcFct(gridY, 'y')

        else:
            ax = self._srcFct(gridX, 'x')
            ay = self._srcFct(gridY, 'y')
            az = self._srcFct(gridZ, 'z')
            a = np.concatenate((ax, ay, az))

        return a

    def _bSrc(self, prob):
        if prob._formulation == 'EB':
            C = prob.mesh.edgeCurl

        elif prob._formulation == 'HJ':
            C = prob.mesh.edgeCurl.T

        return C*self._aSrc(prob)

    def bInitial(self, prob):

        if self.waveform.hasInitialFields is False:
            return Zero()

        return self._bSrc(prob)

    def hInitial(self, prob):

        if self.waveform.hasInitialFields is False:
            return Zero()

        return 1./self.mu * self._bSrc(prob)

    def s_m(self, prob, time):
        if self.waveform.hasInitialFields is False:
            # raise NotImplementedError
            return Zero()
        return Zero()

    def s_e(self, prob, time):
        C = prob.mesh.edgeCurl
        b = self._bSrc(prob)

        if prob._formulation == 'EB':

            MfMui = prob.MfMui

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

        elif prob._formulation == 'HJ':

            h = 1./self.mu * b

            if self.waveform.hasInitialFields is True and time < prob.timeSteps[1]:
                # if time > 0.0:
                #     return Zero()
                if prob._fieldType == 'h':
                    return Zero()
                elif prob._fieldType == 'j':
                    # Compute s_e from vector potential
                    return C * h
            else:
                # b = self._bfromVectorPotential(prob)
                return C * h * self.waveform.eval(time)


class CircularLoop(MagDipole):

    radius = properties.Float(
        "radius of the loop source", default=1., min=0.
    )
    # waveform = None
    # loc = None
    # orientation = 'Z'
    # radius = None
    # mu = mu_0

    def __init__(self, rxList, **kwargs):
        # assert(self.orientation in ['X', 'Y', 'Z']), (
        #     "Orientation (right now) doesn't actually do anything! The methods"
        #     " in SrcUtils should take care of this..."
        #     )
        # self.integrate = False
        BaseTDEMSrc.__init__(self, rxList, **kwargs)

    def _srcFct(self, obsLoc, component):
        return MagneticLoopVectorPotential(
            self.loc, obsLoc, component, mu=self.mu, radius=self.radius
        )


class LineCurrent(BaseTDEMSrc):
    """
    RawVec electric source. It is defined by the user provided vector s_e

    :param list rxList: receiver list
    :param bool integrate: Integrate the source term (multiply by Me) [False]
    """
    waveform = None
    loc = None
    mu = mu_0
    srcType = "Galvanic"

    def __init__(self, rxList, **kwargs):
        self.integrate = False
        BaseEMSrc.__init__(self, rxList, **kwargs)

    def Mejs(self, prob):
        if getattr(self, '_Mejs', None) is None:
            x0 = prob.mesh.x0
            hx = prob.mesh.hx
            hy = prob.mesh.hy
            hz = prob.mesh.hz
            px = self.loc[:, 0]
            py = self.loc[:, 1]
            pz = self.loc[:, 2]
            self._Mejs = getSourceTermLineCurrentPolygon(x0, hx, hy, hz,
                                                         px, py, pz)
        return self._Mejs

    def getRHSdc(self, prob):
        Grad = prob.mesh.nodalGrad
        return Grad.T*self.Mejs(prob)

    # TODO: Need to implement solving MMR for this when
    # StepOffwaveforme is used.
    def bInitial(self, prob):
        if self.waveform.eval(0) == 1.:
            raise Exception("Not implemetned for computing b!")
        else:
            return Zero()

    def eInitial(self, prob):
        if self.waveform.hasInitialFields:
            RHSdc = self.getRHSdc(prob)
            soldc = prob.Adcinv * RHSdc
            return - prob.mesh.nodalGrad * soldc
        else:
            return Zero()

    def eInitialDeriv(self, prob, v=None, adjoint=False, f=None):
        if self.waveform.hasInitialFields:
            edc = f[self, 'e', 0]
            Grad = prob.mesh.nodalGrad
            if adjoint is False:
                AdcDeriv_v = prob.getAdcDeriv(edc, v, adjoint=adjoint)
                edcDeriv = Grad * (prob.Adcinv * AdcDeriv_v)
                return edcDeriv
            elif adjoint is True:
                vec = prob.Adcinv * (Grad.T * v)
                edcDerivT = prob.getAdcDeriv(edc, vec, adjoint=adjoint)
                return edcDerivT
        else:
            return Zero()

    def s_m(self, prob, time):
        return Zero()

    def s_e(self, prob, time):
        return self.Mejs(prob) * self.waveform.eval(time)
