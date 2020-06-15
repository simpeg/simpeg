from __future__ import division, print_function

import numpy as np
from scipy.constants import mu_0
import properties
import warnings
from ...utils.code_utils import deprecate_property

from geoana.em.static import MagneticDipoleWholeSpace, CircularLoopWholeSpace

from ..base import BaseEMSrc
from ..utils import getSourceTermLineCurrentPolygon
from ...props import LocationVector
from ...utils import setKwargs, sdiag, Zero, Identity

###############################################################################
#                                                                             #
#                           Source Waveforms                                  #
#                                                                             #
###############################################################################


class BaseWaveform(properties.HasProperties):

    hasInitialFields = properties.Bool(
        "Does the waveform have initial fields?", default=False
    )

    offTime = properties.Float("off-time of the source", default=0.0)

    eps = properties.Float(
        "window of time within which the waveform is considered on", default=1e-9
    )

    def __init__(self, **kwargs):
        setKwargs(self, **kwargs)

    def eval(self, time):
        raise NotImplementedError

    def evalDeriv(self, time):
        raise NotImplementedError  # needed for E-formulation


class StepOffWaveform(BaseWaveform):
    def __init__(self, offTime=0.0):
        BaseWaveform.__init__(self, offTime=offTime, hasInitialFields=True)

    def eval(self, time):
        if abs(time - 0.0) < self.eps:
            return 1.0
        else:
            return 0.0


class RampOffWaveform(BaseWaveform):
    def __init__(self, offTime=0.0):
        BaseWaveform.__init__(self, offTime=offTime, hasInitialFields=True)

    def eval(self, time):
        if abs(time - 0.0) < self.eps:
            return 1.0
        elif time < self.offTime:
            return -1.0 / self.offTime * (time - self.offTime)
        else:
            return 0.0


class RawWaveform(BaseWaveform):
    def __init__(self, offTime=0.0, waveFct=None, **kwargs):
        self.waveFct = waveFct
        BaseWaveform.__init__(self, offTime=offTime, **kwargs)

    def eval(self, time):
        return self.waveFct(time)


class VTEMWaveform(BaseWaveform):

    offTime = properties.Float("off-time of the source", default=4.2e-3)

    peakTime = properties.Float(
        "Time at which the VTEM waveform is at its peak", default=2.73e-3
    )

    a = properties.Float(
        "parameter controlling how quickly the waveform ramps on", default=3.0
    )

    def __init__(self, **kwargs):
        BaseWaveform.__init__(self, hasInitialFields=False, **kwargs)

    def eval(self, time):
        if time <= self.peakTime:
            return (1.0 - np.exp(-self.a * time / self.peakTime)) / (
                1.0 - np.exp(-self.a)
            )
        elif (time < self.offTime) and (time > self.peakTime):
            return -1.0 / (self.offTime - self.peakTime) * (time - self.offTime)
        else:
            return 0.0


class TrapezoidWaveform(BaseWaveform):
    """
    A waveform that has a linear ramp-on and a linear ramp-off.
    """

    ramp_on = properties.Array(
        """times over which the transmitter ramps on
        [time starting to ramp on, time fully on]
        """,
        shape=(2,),
        dtype=float,
    )

    ramp_off = properties.Array(
        """times over which we ramp off the waveform
        [time starting to ramp off, time off]
        """,
        shape=(2,),
        dtype=float,
    )

    def __init__(self, **kwargs):
        super(TrapezoidWaveform, self).__init__(**kwargs)
        self.hasInitialFields = False

    def eval(self, time):
        if time < self.ramp_on[0]:
            return 0
        elif time >= self.ramp_on[0] and time <= self.ramp_on[1]:
            return (1.0 / (self.ramp_on[1] - self.ramp_on[0])) * (
                time - self.ramp_on[0]
            )
        elif time > self.ramp_on[1] and time < self.ramp_off[0]:
            return 1
        elif time >= self.ramp_off[0] and time <= self.ramp_off[1]:
            return 1 - (1.0 / (self.ramp_off[1] - self.ramp_off[0])) * (
                time - self.ramp_off[0]
            )
        else:
            return 0


class TriangularWaveform(TrapezoidWaveform):
    """
    TriangularWaveform is a special case of TrapezoidWaveform where there's no pleateau
    """

    offTime = properties.Float("off-time of the source")
    peakTime = properties.Float("Time at which the Triangular waveform is at its peak")

    def __init__(self, **kwargs):
        super(TriangularWaveform, self).__init__(**kwargs)
        self.hasInitialFields = False
        self.ramp_on = np.r_[0.0, self.peakTime]
        self.ramp_off = np.r_[self.peakTime, self.offTime]


class QuarterSineRampOnWaveform(BaseWaveform):
    """
    A waveform that has a quarter-sine ramp-on and a linear ramp-off
    """

    ramp_on = properties.Array(
        "times over which the transmitter ramps on", shape=(2,), dtype=float
    )

    ramp_off = properties.Array(
        "times over which we ramp off the waveform", shape=(2,), dtype=float
    )

    def __init__(self, **kwargs):
        super(QuarterSineRampOnWaveform, self).__init__(**kwargs)
        self.hasInitialFields = False

    def eval(self, time):
        if time < self.ramp_on[0]:
            return 0
        elif time >= self.ramp_on[0] and time <= self.ramp_on[1]:
            return np.sin(
                np.pi
                / 2
                * (1.0 / (self.ramp_on[1] - self.ramp_on[0]))
                * (time - self.ramp_on[0])
            )
        elif time > self.ramp_on[1] and time < self.ramp_off[0]:
            return 1
        elif time >= self.ramp_off[0] and time <= self.ramp_off[1]:
            return 1 - (1.0 / (self.ramp_off[1] - self.ramp_off[0])) * (
                time - self.ramp_off[0]
            )
        else:
            return 0


class HalfSineWaveform(BaseWaveform):
    """
    A waveform that has a quarter-sine ramp-on and a quarter-cosine ramp-off.
    When the end of ramp-on and start of ramp off are on the same spot, it looks
    like a half sine wave.
    """

    ramp_on = properties.Array(
        "times over which the transmitter ramps on", shape=(2,), dtype=float
    )
    ramp_off = properties.Array(
        "times over which we ramp off the waveform", shape=(2,), dtype=float
    )

    def __init__(self, **kwargs):
        super(HalfSineWaveform, self).__init__(**kwargs)
        self.hasInitialFields = False

    def eval(self, time):
        if time < self.ramp_on[0]:
            return 0
        elif time >= self.ramp_on[0] and time <= self.ramp_on[1]:
            return np.sin(
                (np.pi / 2)
                * ((time - self.ramp_on[0]) / (self.ramp_on[1] - self.ramp_on[0]))
            )
        elif time > self.ramp_on[1] and time < self.ramp_off[0]:
            return 1
        elif time >= self.ramp_off[0] and time <= self.ramp_off[1]:
            return np.cos(
                (np.pi / 2)
                * ((time - self.ramp_off[0]) / (self.ramp_off[1] - self.ramp_off[0]))
            )
        else:
            return 0


###############################################################################
#                                                                             #
#                                    Sources                                  #
#                                                                             #
###############################################################################


class BaseTDEMSrc(BaseEMSrc):

    # rxPair = Rx
    waveform = properties.Instance(
        "A source waveform", BaseWaveform, default=StepOffWaveform()
    )
    srcType = properties.StringChoice(
        "is the source a galvanic of inductive source",
        choices=["inductive", "galvanic"],
    )

    def __init__(self, receiver_list=None, **kwargs):
        if receiver_list is not None:
            kwargs["receiver_list"] = receiver_list
        super(BaseTDEMSrc, self).__init__(**kwargs)

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
                self.s_eDeriv(prob, time, v, adjoint),
            )
        else:
            return (
                lambda v: self.s_mDeriv(prob, time, v, adjoint),
                lambda v: self.s_eDeriv(prob, time, v, adjoint),
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

    moment = properties.Float("dipole moment of the transmitter", default=1.0, min=0.0)
    mu = properties.Float("permeability of the background", default=mu_0, min=0.0)
    orientation = properties.Vector3(
        "orientation of the source", default="Z", length=1.0, required=True
    )
    location = LocationVector(
        "location of the source", default=np.r_[0.0, 0.0, 0.0], shape=(3,)
    )
    loc = deprecate_property(
        location, "loc", new_name="location", removal_version="0.15.0"
    )

    def __init__(self, receiver_list=None, **kwargs):
        kwargs.pop("srcType", None)
        BaseTDEMSrc.__init__(
            self, receiver_list=receiver_list, srcType="inductive", **kwargs
        )

    def _srcFct(self, obsLoc, coordinates="cartesian"):
        if getattr(self, "_dipole", None) is None:
            self._dipole = MagneticDipoleWholeSpace(
                mu=self.mu,
                orientation=self.orientation,
                location=self.loc,
                moment=self.moment,
            )
        return self._dipole.vector_potential(obsLoc, coordinates=coordinates)

    def _aSrc(self, prob):
        coordinates = "cartesian"
        if prob._formulation == "EB":
            gridX = prob.mesh.gridEx
            gridY = prob.mesh.gridEy
            gridZ = prob.mesh.gridEz

        elif prob._formulation == "HJ":
            gridX = prob.mesh.gridFx
            gridY = prob.mesh.gridFy
            gridZ = prob.mesh.gridFz

        if prob.mesh._meshType == "CYL":
            coordinates = "cylindrical"
            if prob.mesh.isSymmetric:
                return self._srcFct(gridY)[:, 1]

        ax = self._srcFct(gridX, coordinates)[:, 0]
        ay = self._srcFct(gridY, coordinates)[:, 1]
        az = self._srcFct(gridZ, coordinates)[:, 2]
        a = np.concatenate((ax, ay, az))

        return a

    def _getAmagnetostatic(self, prob):
        if prob._formulation == "EB":
            return prob.mesh.faceDiv * prob.MfMuiI * prob.mesh.faceDiv.T
        else:
            raise NotImplementedError(
                "Solving the magnetostatic problem for the initial fields "
                "when a permeable model is considered has not yet been "
                "implemented for the HJ formulation. "
                "See: https://github.com/simpeg/simpeg/issues/680"
            )

    def _rhs_magnetostatic(self, prob):
        if getattr(self, "_hp", None) is None:
            if prob._formulation == "EB":
                bp = prob.mesh.edgeCurl * self._aSrc(prob)
                self._MfMuip = prob.mesh.getFaceInnerProduct(1.0 / self.mu)
                self._MfMuipI = prob.mesh.getFaceInnerProduct(
                    1.0 / self.mu, invMat=True
                )
                self._hp = self._MfMuip * bp
            else:
                raise NotImplementedError(
                    "Solving the magnetostatic problem for the initial fields "
                    "when a permeable model is considered has not yet been "
                    "implemented for the HJ formulation. "
                    "See: https://github.com/simpeg/simpeg/issues/680"
                )

        if prob._formulation == "EB":
            return -prob.mesh.faceDiv * ((prob.MfMuiI - self._MfMuipI) * self._hp)
        else:
            raise NotImplementedError(
                "Solving the magnetostatic problem for the initial fields "
                "when a permeable model is considered has not yet been "
                "implemented for the HJ formulation. "
                "See: https://github.com/simpeg/simpeg/issues/680"
            )

    def _phiSrc(self, prob):
        Ainv = prob.Solver(self._getAmagnetostatic(prob))  # todo: store these
        rhs = self._rhs_magnetostatic(prob)
        Ainv.clean()
        return Ainv * rhs

    def _bSrc(self, prob):
        if prob._formulation == "EB":
            C = prob.mesh.edgeCurl

        elif prob._formulation == "HJ":
            C = prob.mesh.edgeCurl.T

        return C * self._aSrc(prob)

    def bInitial(self, prob):

        if self.waveform.hasInitialFields is False:
            return Zero()

        if np.all(prob.mu == self.mu):
            return self._bSrc(prob)

        else:
            if prob._formulation == "EB":
                hs = prob.mesh.faceDiv.T * self._phiSrc(prob)
                ht = self._hp + hs
                return prob.MfMuiI * ht
            else:
                raise NotImplementedError

    def hInitial(self, prob):

        if self.waveform.hasInitialFields is False:
            return Zero()
        # if prob._formulation == 'EB':
        #     return prob.MfMui * self.bInitial(prob)
        # elif prob._formulation == 'HJ':
        #     return prob.MeMuI * self.bInitial(prob)
        return 1.0 / self.mu * self.bInitial(prob)

    def s_m(self, prob, time):
        if self.waveform.hasInitialFields is False:
            return Zero()
        return Zero()

    def s_e(self, prob, time):
        C = prob.mesh.edgeCurl
        b = self._bSrc(prob)

        if prob._formulation == "EB":
            MfMui = prob.mesh.getFaceInnerProduct(1.0 / self.mu)

            if self.waveform.hasInitialFields is True and time < prob.time_steps[1]:
                if prob._fieldType == "b":
                    return Zero()
                elif prob._fieldType == "e":
                    # Compute s_e from vector potential
                    return C.T * (MfMui * b)
            else:
                return C.T * (MfMui * b) * self.waveform.eval(time)

        elif prob._formulation == "HJ":

            h = 1.0 / self.mu * b

            if self.waveform.hasInitialFields is True and time < prob.time_steps[1]:
                if prob._fieldType == "h":
                    return Zero()
                elif prob._fieldType == "j":
                    # Compute s_e from vector potential
                    return C * h
            else:
                return C * h * self.waveform.eval(time)


class CircularLoop(MagDipole):

    radius = properties.Float("radius of the loop source", default=1.0, min=0.0)

    current = properties.Float("current in the loop", default=1.0)

    N = properties.Float("number of turns in the loop", default=1.0)

    def __init__(self, receiver_list=None, **kwargs):
        super(CircularLoop, self).__init__(receiver_list, **kwargs)

    @property
    def moment(self):
        return np.pi * self.radius ** 2 * self.current * self.N

    def _srcFct(self, obsLoc, coordinates="cartesian"):
        # return MagneticLoopVectorPotential(
        #     self.loc, obsLoc, component, mu=self.mu, radius=self.radius
        # )

        if getattr(self, "_loop", None) is None:
            self._loop = CircularLoopWholeSpace(
                mu=self.mu,
                location=self.loc,
                orientation=self.orientation,
                radius=self.radius,
                current=self.current,
            )
        return self._loop.vector_potential(obsLoc, coordinates)


class LineCurrent(BaseTDEMSrc):
    """
    RawVec electric source. It is defined by the user provided vector s_e

    :param list rxList: receiver list
    :param bool integrate: Integrate the source term (multiply by Me) [False]
    """

    location = properties.Array("location of the source", shape=("*", 3))
    loc = deprecate_property(
        location, "loc", new_name="location", removal_version="0.15.0"
    )

    def __init__(self, receiver_list=None, **kwargs):
        self.integrate = False
        kwargs.pop("srcType", None)
        super(LineCurrent, self).__init__(receiver_list, srcType="galvanic", **kwargs)

    def Mejs(self, prob):
        if getattr(self, "_Mejs", None) is None:
            x0 = prob.mesh.x0
            hx = prob.mesh.hx
            hy = prob.mesh.hy
            hz = prob.mesh.hz
            px = self.loc[:, 0]
            py = self.loc[:, 1]
            pz = self.loc[:, 2]
            self._Mejs = getSourceTermLineCurrentPolygon(x0, hx, hy, hz, px, py, pz)
        return self._Mejs

    def getRHSdc(self, prob):
        Grad = prob.mesh.nodalGrad
        return Grad.T * self.Mejs(prob)

    # TODO: Need to implement solving MMR for this when
    # StepOffwaveform is used.
    def bInitial(self, prob):
        if self.waveform.eval(0) == 1.0:
            raise Exception("Not implemetned for computing b!")
        else:
            return Zero()

    def eInitial(self, prob):
        if self.waveform.hasInitialFields:
            RHSdc = self.getRHSdc(prob)
            soldc = prob.Adcinv * RHSdc
            return -prob.mesh.nodalGrad * soldc
        else:
            return Zero()

    def jInitial(self, prob):
        raise NotImplementedError

    def hInitial(self, prob):
        raise NotImplementedError

    def eInitialDeriv(self, prob, v=None, adjoint=False, f=None):
        if self.waveform.hasInitialFields:
            edc = f[self, "e", 0]
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


# TODO: this should be generalized and plugged into getting the Line current
# on faces
class RawVec_Grounded(BaseTDEMSrc):

    # mu = properties.Float(
    #     "permeability of the background", default=mu_0, min=0.
    # )

    _s_e = properties.Array("source term", shape=("*",))

    def __init__(self, receiver_list=None, s_e=None, **kwargs):
        self.integrate = False
        kwargs.pop("srcType", None)
        super(RawVec_Grounded, self).__init__(
            receiver_list, srcType="galvanic", **kwargs
        )

        if s_e is not None:
            self._s_e = s_e

    def getRHSdc(self, prob):
        return sdiag(prob.mesh.vol) * prob.mesh.faceDiv * self._s_e

    def phiInitial(self, prob):
        if self.waveform.hasInitialFields:
            RHSdc = self.getRHSdc(prob)
            return prob.Adcinv * RHSdc
        else:
            return Zero()

    def _phiInitialDeriv(self, prob, v, adjoint=False):
        if self.waveform.hasInitialFields:
            phi = self.phiInitial(prob)

            if adjoint is True:
                return -1.0 * prob.getAdcDeriv(
                    phi, prob.Adcinv * v, adjoint=True
                )  # A is symmetric

            Adc_deriv = prob.getAdcDeriv(phi, v)
            return -1.0 * (prob.Adcinv * Adc_deriv)

        else:
            return Zero()

    def jInitial(self, prob):
        if prob._fieldType not in ["j", "h"]:
            raise NotImplementedError

        if self.waveform.hasInitialFields is False:
            return Zero()

        phi = self.phiInitial(prob)
        Div = sdiag(prob.mesh.vol) * prob.mesh.faceDiv
        return -prob.MfRhoI * (Div.T * phi)

    def jInitialDeriv(self, prob, v, adjoint=False):
        if prob._fieldType not in ["j", "h"]:
            raise NotImplementedError

        if self.waveform.hasInitialFields is False:
            return Zero()

        phi = self.phiInitial(prob)
        Div = sdiag(prob.mesh.vol) * prob.mesh.faceDiv

        if adjoint is True:
            return -(
                prob.MfRhoIDeriv(Div.T * phi, v=v, adjoint=True)
                + self._phiInitialDeriv(prob, Div * (prob.MfRhoI.T * v), adjoint=True)
            )
        phiDeriv = self._phiInitialDeriv(prob, v)
        return -(prob.MfRhoIDeriv(Div.T * phi, v=v) + prob.MfRhoI * (Div.T * phiDeriv))

    def _getAmmr(self, prob):
        if prob._fieldType not in ["j", "h"]:
            raise NotImplementedError

        vol = prob.mesh.vol

        return (
            prob.mesh.edgeCurl * prob.MeMuI * prob.mesh.edgeCurl.T
            - prob.mesh.faceDiv.T
            * sdiag(1.0 / vol * prob.mui)
            * prob.mesh.faceDiv  # stabalizing term. See (Chen, Haber & Oldenburg 2002)
        )

    def _aInitial(self, prob):
        A = self._getAmmr(prob)
        Ainv = prob.Solver(A)  # todo: store this
        s_e = self.s_e(prob, 0)
        rhs = s_e - self.jInitial(prob)
        return Ainv * rhs

    def _aInitialDeriv(self, prob, v, adjoint=False):
        A = self._getAmmr(prob)
        Ainv = prob.Solver(A)  # todo: store this - move it to the problem

        if adjoint is True:
            return -1 * (
                self.jInitialDeriv(prob, Ainv * v, adjoint=True)
            )  # A is symmetric

        return -1 * (Ainv * self.jInitialDeriv(prob, v))

    def hInitial(self, prob):
        if prob._fieldType not in ["j", "h"]:
            raise NotImplementedError

        if self.waveform.hasInitialFields is False:
            return Zero()

        b = self.bInitial(prob)
        return prob.MeMuI * b

    def hInitialDeriv(self, prob, v, adjoint=False, f=None):
        if prob._fieldType not in ["j", "h"]:
            raise NotImplementedError

        if self.waveform.hasInitialFields is False:
            return Zero()

        if adjoint is True:
            return self.bInitialDeriv(prob, prob.MeMuI.T * v, adjoint=True)
        return prob.MeMuI * self.bInitialDeriv(prob, v)

    def bInitial(self, prob):
        if prob._fieldType not in ["j", "h"]:
            raise NotImplementedError

        if self.waveform.hasInitialFields is False:
            return Zero()

        a = self._aInitial(prob)
        return prob.mesh.edgeCurl.T * a

    def bInitialDeriv(self, prob, v, adjoint=False, f=None):
        if prob._fieldType not in ["j", "h"]:
            raise NotImplementedError

        if self.waveform.hasInitialFields is False:
            return Zero()

        if adjoint is True:
            return self._aInitialDeriv(prob, prob.mesh.edgeCurl * v, adjoint=True)
        return prob.mesh.edgeCurl.T * self._aInitialDeriv(prob, v)

    def s_e(self, prob, time):
        # if prob._fieldType == 'h':
        #     return prob.Mf * self._s_e * self.waveform.eval(time)
        return self._s_e * self.waveform.eval(time)
