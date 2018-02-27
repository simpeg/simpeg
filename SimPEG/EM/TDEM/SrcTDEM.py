from __future__ import division, print_function

import numpy as np
from scipy.constants import mu_0
import properties
import warnings
import types

from ...Utils import Zero, Identity
from ..Utils import (
    MagneticDipoleVectorPotential, getSourceTermLineCurrentPolygon,
    MagneticLoopVectorPotential
)
from ..Base import BaseEMSrc


###############################################################################
#                                                                             #
#                           Source Waveforms                                  #
#                                                                             #
###############################################################################

class BaseWaveform(properties.HasProperties):

    # TODO: _has_initial_fields
    hasInitialFields = properties.Bool(
        "Does the waveform have initial fields?", default=False
    )

    # TODO: off_time
    offTime = properties.Float(
        "offTime of the source", default=0.
    )

    # TODO: this should be private
    eps = properties.Float(
        "window of time within which the waveform is considered on",
        default=1e-9
    )

    def __init__(self, **kwargs):
        super(BaseWaveform, self).__init__(**kwargs)

    def _assertMatchesPair(self, pair):
        assert isinstance(self, pair), (
            "Waveform object must be an instance of a %s "
            "BaseWaveform class.".format(pair.__name__)
        )

    # TODO: call
    def eval(self, time):
        raise NotImplementedError

    # TODO: deriv
    def evalDeriv(self, time):
        raise NotImplementedError  # needed for E-formulation


class StepOffWaveform(BaseWaveform):

    def __init__(self, **kwargs):
        super(StepOffWaveform, self).__init__(**kwargs)
        self.hasInitialFields = True

    def eval(self, time):
        if abs(time-0.) < self.eps:
            return 1.
        else:
            return 0.


class RampOffWaveform(BaseWaveform):

    def __init__(self, **kwargs):
        super(RampOffWaveform, self).__init__(**kwargs)
        self.hasInitialFields = True

    def eval(self, time):
        if abs(time-0.) < self.eps:
            return 1.
        elif time < self.offTime:
            return -1. / self.offTime * (time - self.offTime)
        else:
            return 0.


class RawWaveform(BaseWaveform):

    # todo: think about how to turn this into a property
    waveFct = None  #: wave function

    def __init__(self, **kwargs):
        super(RawWaveform, self).__init__(**kwargs)

    def eval(self, time):
        return self.waveFct(time)


class TriangularWaveform(BaseWaveform):

    def __init__(self, **kwargs):
        BaseWaveform.__init__(self, **kwargs)
        self.hasInitialFields = True

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
        super(VTEMWaveform, self).__init__(self, **kwargs)
        self.hasInitialFields = False

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
#                            Base TDEM Source                                 #
#                                                                             #
###############################################################################

class BaseTDEMSrc(BaseEMSrc):
    """
    Base class for a time domain EM source. Inherit this to build your own
    TDEM source
    """

    waveform = properties.Instance(
        "a TDEM waveform instance",
        BaseWaveform,
        default=StepOffWaveform()
    )

    # TODO: src_type
    srcType = properties.StringChoice(
        "type of the source. Is it grounded or inductive?",
        choices={
            'grounded': ['galvanic'],
            'inductive': ['loop', 'magnetic']
        }
    )

    def __init__(self, **kwargs):
        super(BaseTDEMSrc, self).__init__(**kwargs)

    def bInitial(self, simulation):
        return Zero()

    def bInitialDeriv(self, simulation, v=None, adjoint=False, f=None):
        return Zero()

    def eInitial(self, simulation):
        return Zero()

    def eInitialDeriv(self, simulation, v=None, adjoint=False, f=None):
        return Zero()

    def hInitial(self, simulation):
        return Zero()

    def hInitialDeriv(self, simulation, v=None, adjoint=False, f=None):
        return Zero()

    def jInitial(self, simulation):
        return Zero()

    def jInitialDeriv(self, simulation, v=None, adjoint=False, f=None):
        return Zero()

    def eval(self, simulation, time):
        s_m = self.s_m(simulation, time)
        s_e = self.s_e(simulation, time)
        return s_m, s_e

    def evalDeriv(self, simulation, time, v=None, adjoint=False):
        if v is not None:
            return (
                self.s_mDeriv(simulation, time, v, adjoint),
                self.s_eDeriv(simulation, time, v, adjoint)
            )
        else:
            return (
                lambda v: self.s_mDeriv(simulation, time, v, adjoint),
                lambda v: self.s_eDeriv(simulation, time, v, adjoint)
            )

    def s_m(self, simulation, time):
        return Zero()

    def s_e(self, simulation, time):
        return Zero()

    def s_mDeriv(self, simulation, time, v=None, adjoint=False):
        return Zero()

    def s_eDeriv(self, simulation, time, v=None, adjoint=False):
        return Zero()


###############################################################################
#                                                                             #
#                                    Sources                                  #
#                                                                             #
###############################################################################

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

    def __init__(self, **kwargs):
        # assert(self.orientation in ['X', 'Y', 'Z']), (
        #     "Orientation (right now) doesn't actually do anything! The methods"
        #     " in SrcUtils should take care of this..."
        #     )
        # self.integrate = False
        super(MagDipole, self).__init__(**kwargs)
        getSourceTermLineCurrentPolygon

    @properties.validator('orientation')
    def _warn_non_axis_aligned_sources(self, change):
        value = change['value']
        axaligned = [
            True for vec in [
                np.r_[1., 0., 0.], np.r_[0., 1., 0.], np.r_[0., 0., 1.]
            ]
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

    def _aSrc(self, simulation):
        if simulation._formulation == 'EB':
            gridX = simulation.mesh.gridEx
            gridY = simulation.mesh.gridEy
            gridZ = simulation.mesh.gridEz

        elif simulation._formulation == 'HJ':
            gridX = simulation.mesh.gridFx
            gridY = simulation.mesh.gridFy
            gridZ = simulation.mesh.gridFz

        if simulation.mesh._meshType is 'CYL':
            if not simulation.mesh.isSymmetric:
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

    def _bSrc(self, simulation):
        if simulation._formulation == 'EB':
            C = simulation.mesh.edgeCurl

        elif simulation._formulation == 'HJ':
            C = simulation.mesh.edgeCurl.T

        return C*self._aSrc(simulation)

    def bInitial(self, simulation):

        if self.waveform.hasInitialFields is False:
            return Zero()

        return self._bSrc(simulation)

    def hInitial(self, simulation):

        if self.waveform.hasInitialFields is False:
            return Zero()

        return 1./self.mu * self._bSrc(simulation)

    def s_m(self, simulation, time):
        if self.waveform.hasInitialFields is False:
            # raise NotImplementedError
            return Zero()
        return Zero()

    def s_e(self, simulation, time):
        C = simulation.mesh.edgeCurl
        b = self._bSrc(simulation)

        if simulation._formulation == 'EB':

            MfMui = simulation.MfMui

            if self.waveform.hasInitialFields is True and time < simulation.time_steps[1]:
                # if time > 0.0:
                #     return Zero()
                if simulation._fieldType == 'b':
                    return Zero()
                elif simulation._fieldType == 'e':
                    # Compute s_e from vector potential
                    return C.T * (MfMui * b)
            else:
                # b = self._bfromVectorPotential(simulation)
                return C.T * (MfMui * b) * self.waveform.eval(time)
        # return Zero()

        elif simulation._formulation == 'HJ':

            h = 1./self.mu * b

            if self.waveform.hasInitialFields is True and time < simulation.time_steps[1]:
                # if time > 0.0:
                #     return Zero()
                if simulation._fieldType == 'h':
                    return Zero()
                elif simulation._fieldType == 'j':
                    # Compute s_e from vector potential
                    return C * h
            else:
                # b = self._bfromVectorPotential(simulation)
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

    def __init__(self, **kwargs):
        # assert(self.orientation in ['X', 'Y', 'Z']), (
        #     "Orientation (right now) doesn't actually do anything! The methods"
        #     " in SrcUtils should take care of this..."
        #     )
        # self.integrate = False
        super(CircularLoop, self).__init__(**kwargs)

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
    # waveform = None
    loc = properties.Array(
        "locations of the two electrodes, np.vstack([Aloc, Bloc])",
        shape=(2, 3),
        dtype=float
    )
    # mu = mu_0

    def __init__(self, **kwargs):
        super(LineCurrent, self).__init__(**kwargs)
        self.integrate = False
        self.srcType = 'grounded'

    def Mejs(self, simulation):
        if getattr(self, '_Mejs', None) is None:
            x0 = simulation.mesh.x0
            hx = simulation.mesh.hx
            hy = simulation.mesh.hy
            hz = simulation.mesh.hz
            px = self.loc[:, 0]
            py = self.loc[:, 1]
            pz = self.loc[:, 2]
            self._Mejs = getSourceTermLineCurrentPolygon(
                x0, hx, hy, hz, px, py, pz
            )
        return self._Mejs

    def getRHSdc(self, simulation):
        Grad = simulation.mesh.nodalGrad
        return Grad.T*self.Mejs(simulation)

    # TODO: Need to implement solving MMR for this when
    # StepOffwaveforme is used.
    def bInitial(self, simulation):
        if self.waveform.eval(0) == 1.:
            raise NotImplementedError(
                "bInitial has not been implemented for computing b"
            )
        else:
            return Zero()

    def eInitial(self, simulation):
        if self.waveform.hasInitialFields:
            RHSdc = self.getRHSdc(simulation)
            soldc = simulation.Adcinv * RHSdc
            return - simulation.mesh.nodalGrad * soldc
        else:
            return Zero()

    def eInitialDeriv(self, simulation, v=None, adjoint=False, f=None):
        if self.waveform.hasInitialFields:
            edc = f[self, 'e', 0]
            Grad = simulation.mesh.nodalGrad
            if adjoint is False:
                AdcDeriv_v = simulation.getAdcDeriv(edc, v, adjoint=adjoint)
                edcDeriv = Grad * (simulation.Adcinv * AdcDeriv_v)
                return edcDeriv
            elif adjoint is True:
                vec = simulation.Adcinv * (Grad.T * v)
                edcDerivT = simulation.getAdcDeriv(edc, vec, adjoint=adjoint)
                return edcDerivT
        else:
            return Zero()

    def s_m(self, simulation, time):
        return Zero()

    def s_e(self, simulation, time):
        return self.Mejs(simulation) * self.waveform.eval(time)
