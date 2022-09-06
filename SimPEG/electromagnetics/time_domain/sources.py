import warnings

import numpy as np
import properties
from geoana.em.static import CircularLoopWholeSpace, MagneticDipoleWholeSpace
from scipy.constants import mu_0

from ...props import LocationVector
from ...utils import Zero, sdiag, setKwargs
from ...utils.code_utils import deprecate_property
from ..base import BaseEMSrc
from ..utils import line_through_faces, segmented_line_current_source_term

###############################################################################
#                                                                             #
#                           Source Waveforms                                  #
#                                                                             #
###############################################################################


class BaseWaveform:
    """
    Base class for creating a waveform for time-domain EM simulations.

    Parameters
    ----------
    has_initial_fields: bool
        If the transmitter has non-zero current prior to the start of the simulation
        (e.g. a step-off waveform), set `has_initial_fields` to True

    off_time: float
        Time when the transmitter current is zero in units of seconds. Default is 0.0

    epsilon: float
        Small time-constant for which the transmitter is assumed to still be on for
    """

    def __init__(self, has_initial_fields=False, off_time=0.0, epsilon=1e-9, **kwargs):
        self.has_initial_fields = has_initial_fields
        self.off_time = off_time
        self.epsilon = epsilon
        setKwargs(self, **kwargs)

    @property
    def has_initial_fields(self):
        """Does the waveform have initial fields?"""
        return self._has_initial_fields

    @has_initial_fields.setter
    def has_initial_fields(self, value):
        if not isinstance(value, bool):
            raise ValueError(
                "The value of has_initial_fields must be a bool (True / False)."
                f" The provided value, {value} is not."
            )
        else:
            self._has_initial_fields = value

    @property
    def off_time(self):
        return self._off_time

    @off_time.setter
    def off_time(self, value):
        """ "off-time of the source"""
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            raise ValueError(
                f"off_time must be a float, the value provided, {value} is "
                f"{type(value)}"
            )
        self._off_time = value

    @property
    def epsilon(self):
        """window of time within which the waveform is considered on"""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        if not isinstance(value, float):
            raise ValueError(
                f"epsilon must be a float, the value provided, {value} is "
                f"{type(value)}"
            )
        if value < 0:
            raise ValueError(
                f"epsilon must be greater than 0, the value provided, {value} is not"
            )
        self._epsilon = value

    def eval(self, time):
        raise NotImplementedError

    def evalDeriv(self, time):
        raise NotImplementedError  # needed for E-formulation

    ##########################
    # Deprecated
    ##########################
    hasInitialFields = deprecate_property(
        has_initial_fields,
        "hasInitialFields",
        new_name="has_initial_fields",
        removal_version="0.17.0",
        future_warn=True,
    )

    offTime = deprecate_property(
        off_time,
        "offTime",
        new_name="off_time",
        removal_version="0.17.0",
        future_warn=True,
    )

    eps = deprecate_property(
        epsilon,
        "eps",
        new_name="epsilon",
        removal_version="0.17.0",
        future_warn=True,
    )


class StepOffWaveform(BaseWaveform):
    """
    A heavy-side step function waveform. This is the default for time-domain EM simulations.

    Parameters
    ----------
    off_time: float
        time at which the transmitter is turned off in units of seconds (default is 0s)

    Examples
    --------
    The default off-time for the step-off waveform is 0s. In the example below, we set it to
    1e-5s (0.01msec) to illustrate it in a plot

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from SimPEG.electromagnetics import time_domain as tdem

    >>> times = np.linspace(0, 1e-4, 1000)
    >>> waveform = tdem.sources.StepOffWaveform(off_time=1e-5)
    >>> plt.plot(times, [waveform.eval(t) for t in times])
    >>> plt.show()

    """

    def __init__(self, off_time=0.0, **kwargs):
        super(StepOffWaveform, self).__init__(
            off_time=off_time, has_initial_fields=True
        )

    def eval(self, time):
        if (abs(time - 0.0) < self.epsilon) or ((time - self.off_time) < self.epsilon):
            return 1.0
        else:
            return 0.0


class RampOffWaveform(BaseWaveform):
    """
    A waveform with a linear ramp-off.

    Parameters
    ----------
    off_time: float
        time at which the transmitter is turned off in units of seconds (default is 0s)

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from SimPEG.electromagnetics import time_domain as tdem

    >>> times = np.linspace(0, 1e-4, 1000)
    >>> waveform = tdem.sources.RampOffWaveform(off_time=1e-5)
    >>> plt.plot(times, [waveform.eval(t) for t in times])
    >>> plt.show()

    """

    def __init__(self, off_time=0.0, **kwargs):
        BaseWaveform.__init__(
            self, off_time=off_time, has_initial_fields=True, **kwargs
        )

    def eval(self, time):
        if abs(time - 0.0) < self.epsilon:
            return 1.0
        elif time < self.off_time:
            return -1.0 / self.off_time * (time - self.off_time)
        else:
            return 0.0


class RawWaveform(BaseWaveform):
    """
    A waveform you can define. You need to provide a `waveform_function` that returns
    the waveform evaluated at a given time. This can be used, for example if you would
    like to interpolate between points specified in a waveform file.

    Parameters
    ----------
    off_time: float
        time at which the transmitter is turned off in units of seconds (default is 0s)

    waveform_function: function

    Examples
    --------

    In this example, we define a saw-tooth waveform

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from SimPEG.electromagnetics import time_domain as tdem

    >>> def my_waveform(t):
    >>>     period = 1e-2
    >>>     quarter_period = period / 4
    >>>     t_cycle = np.mod(t, period)
    >>>     if t_cycle <= quarter_period:
    >>>         return t_cycle / quarter_period
    >>>     elif (t_cycle > quarter_period) & (t_cycle <= 3*quarter_period):
    >>>         return -t_cycle / quarter_period + 2
    >>>     elif t_cycle > 3*quarter_period:
    >>>         return t_cycle / quarter_period - 4

    >>> times = np.linspace(0, 1e-2, 1000)
    >>> waveform = tdem.sources.RawWaveform(waveform_function=my_waveform)
    >>> plt.plot(times, [waveform.eval(t) for t in times])
    >>> plt.show()

    """

    def __init__(self, off_time=0.0, waveform_function=None, **kwargs):
        super(RawWaveform, self).__init__(off_time=off_time)
        if waveform_function is not None:
            self.waveform_function = waveform_function
        wavefct = kwargs.pop("waveFct", None)
        if wavefct is not None:
            self.waveFct = wavefct
        setKwargs(self, **kwargs)

    @property
    def waveform_function(self):
        return self._waveform_function

    @waveform_function.setter
    def waveform_function(self, value):
        if not callable(value):
            raise ValueError(
                "waveform_function must be a function. The input value is type: "
                f"{type(value)}"
            )
        self._waveform_function = value

    def eval(self, time):
        return self.waveform_function(time)

    waveFct = deprecate_property(
        waveform_function,
        "waveFct",
        new_name="waveform_function",
        removal_version="0.17.0",
        future_warn=True,
    )


class VTEMWaveform(BaseWaveform):
    """
    A VTEM style waveform

    Parameters
    ----------
    off_time: float
        time at which the transmitter is turned off in units of seconds (default is 4.2e-3s)

    peak_time: float
        the peak time for the waveform (default: 2.73e-3)

    ramp_on_rate: float
        parameter controlling how quickly the waveform ramps on (default is 3)

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from SimPEG.electromagnetics import time_domain as tdem

    >>> times = np.linspace(0, 1e-2, 1000)
    >>> waveform = tdem.sources.VTEMWaveform()
    >>> plt.plot(times, [waveform.eval(t) for t in times])
    >>> plt.show()

    """

    def __init__(self, off_time=4.2e-3, peak_time=2.73e-3, ramp_on_rate=3.0, **kwargs):
        BaseWaveform.__init__(self, has_initial_fields=False, off_time=off_time)
        self.peak_time = peak_time
        self.ramp_on_rate = (
            ramp_on_rate  # we should come up with a better name for this
        )
        setKwargs(self, **kwargs)

    @property
    def peak_time(self):
        return self._peak_time

    @peak_time.setter
    def peak_time(self, value):
        if not isinstance(value, float):
            raise ValueError(
                f"peak_time must be a float, the value provided, {value} is "
                f"{type(value)}"
            )
        if value > self.off_time:
            raise ValueError(
                f"peak_time must be less than off_time {self.off_time}. "
                f"The value provided {value} is not"
            )
        self._peak_time = value

    @property
    def ramp_on_rate(self):
        return self._ramp_on_rate

    @ramp_on_rate.setter
    def ramp_on_rate(self, value):
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            raise ValueError(
                f"ramp_on_rate must be a float, the value provided, {value} is "
                f"{type(value)}"
            )
        self._ramp_on_rate = value

    def eval(self, time):
        if time <= self.peak_time:
            return (1.0 - np.exp(-self.ramp_on_rate * time / self.peak_time)) / (
                1.0 - np.exp(-self.ramp_on_rate)
            )
        elif (time < self.off_time) and (time > self.peak_time):
            return -1.0 / (self.off_time - self.peak_time) * (time - self.off_time)
        else:
            return 0.0

    ##########################
    # Deprecated
    ##########################

    peakTime = deprecate_property(
        peak_time,
        "peakTime",
        new_name="peak_time",
        removal_version="0.17.0",
        future_warn=True,
    )

    a = deprecate_property(
        ramp_on_rate,
        "a",
        new_name="ramp_on_rate",
        removal_version="0.17.0",
        future_warn=True,
    )


class TrapezoidWaveform(BaseWaveform):
    """
    A waveform that has a linear ramp-on and a linear ramp-off.

    Parameters
    ----------
    ramp_on: float
        time when the linear ramp_on ends

    ramp_off: float
        start of the ramp_off

    off_time: float
        time when the transmitter_current returns to zero

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from SimPEG.electromagnetics import time_domain as tdem

    >>> times = np.linspace(0, 1e-2, 1000)
    >>> waveform = tdem.sources.TrapezoidWaveform(ramp_on=2e-3, ramp_off=4e-3, off_time=6e-3)
    >>> plt.plot(times, [waveform.eval(t) for t in times])
    >>> plt.show()

    """

    def __init__(self, ramp_on, ramp_off, off_time=None, **kwargs):
        super(TrapezoidWaveform, self).__init__(has_initial_fields=False)
        self.ramp_on = ramp_on
        self.ramp_off = ramp_off
        self.off_time = off_time if off_time is not None else self.ramp_off[-1]
        setKwargs(self, **kwargs)

    @property
    def ramp_on(self):
        """times over which the transmitter ramps on
        [time starting to ramp on, time fully on]
        """
        return self._ramp_on

    @ramp_on.setter
    def ramp_on(self, value):
        if isinstance(value, (tuple, list)):
            value = np.array(value, dtype=float)
        if not isinstance(value, np.ndarray):
            raise ValueError(
                f"ramp_on must be a numpy array, list or tuple, the value provided, {value} is "
                f"{type(value)}"
            )
        if len(value) != 2:
            raise ValueError(
                f"ramp_on must be length 2 [start, end]. The value provided has "
                f"length {len(value)}"
            )

        value = value.astype(float)
        self._ramp_on = value

    @property
    def ramp_off(self):
        """times over which we ramp off the waveform
        [time starting to ramp off, time off]
        """
        return self._ramp_off

    @ramp_off.setter
    def ramp_off(self, value):
        if isinstance(value, (tuple, list)):
            value = np.array(value, dtype=float)
        if not isinstance(value, np.ndarray):
            raise ValueError(
                f"ramp_off must be a numpy array, list or tuple, the value provided, {value} is "
                f"{type(value)}"
            )
        if len(value) != 2:
            raise ValueError(
                f"ramp_off must be length 2 [start, end]. The value provided has "
                f"length {len(value)}"
            )

        value = value.astype(float)
        self._ramp_off = value

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

    Parameters
    ----------
    off_time: float
        time when the transmitter current returns to zero

    peak_time: float
        time when the transmitter waveform is at a peak

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from SimPEG.electromagnetics import time_domain as tdem

    >>> times = np.linspace(0, 1e-2, 1000)
    >>> waveform = tdem.sources.TriangularWaveform(off_time=6e-3, peak_time=3e-3)
    >>> plt.plot(times, [waveform.eval(t) for t in times])
    >>> plt.show()

    """

    def __init__(self, off_time=None, peak_time=None, **kwargs):
        if peak_time is None:
            peak_time = kwargs.get("peakTime")
            if peak_time is None:
                raise Exception("peak_time must be provided")
            else:
                warnings.warn(
                    "peakTime will be deprecated in 0.17.0. Please update your code to use peak_time instead",
                    FutureWarning,
                )

        if off_time is None:
            off_time = kwargs.pop("offTime")
            if off_time is None:
                raise Exception("off_time must be provided")
            else:
                warnings.warn(
                    "offTime will be deprecated in 0.17.0. Please update your code to use off_time instead",
                    FutureWarning,
                )

        ramp_on = np.r_[0.0, peak_time]
        ramp_off = np.r_[peak_time, off_time]

        super(TriangularWaveform, self).__init__(
            off_time=off_time,
            ramp_on=ramp_on,
            ramp_off=ramp_off,
            has_initial_fields=False,
        )
        self.peak_time = peak_time
        setKwargs(self, **kwargs)

    @property
    def peak_time(self):
        return self._peak_time

    @peak_time.setter
    def peak_time(self, value):
        if not isinstance(value, float):
            raise ValueError(
                f"peak_time must be a float, the value provided, {value} is "
                f"{type(value)}"
            )
        if value > self.off_time:
            raise ValueError(
                f"peak_time must be less than off_time {self.off_time}. "
                f"The value provided {value} is not"
            )
        self._peak_time = value
        self._ramp_on = np.r_[self._ramp_on[0], value]
        self._ramp_off = np.r_[value, self._ramp_off[1]]

    ##########################
    # Deprecated
    ##########################

    peakTime = deprecate_property(
        peak_time,
        "peakTime",
        new_name="peak_time",
        removal_version="0.17.0",
        future_warn=True,
    )


class QuarterSineRampOnWaveform(TrapezoidWaveform):
    """
    A waveform that has a quarter-sine ramp-on and a linear ramp-off

    Parameters
    ----------
    ramp_on: tuple
        times during which the transmitter ramps on

    ramp_off: tuple
        times between which there is a linear ramp-off

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from SimPEG.electromagnetics import time_domain as tdem

    >>> times = np.linspace(0, 1e-2, 1000)
    >>> waveform = tdem.sources.QuarterSineRampOnWaveform(ramp_on=(0, 2e-3), ramp_off=(3e-3, 3.5e-3))
    >>> plt.plot(times, [waveform.eval(t) for t in times])
    >>> plt.show()
    """

    def __init__(self, ramp_on, ramp_off, **kwargs):
        super(QuarterSineRampOnWaveform, self).__init__(
            ramp_on=ramp_on, ramp_off=ramp_off, **kwargs
        )

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


class HalfSineWaveform(TrapezoidWaveform):
    """
    A waveform that has a quarter-sine ramp-on and a quarter-cosine ramp-off.
    When the end of ramp-on and start of ramp off are on the same spot, it looks
    like a half sine wave.
    """

    def __init__(self, ramp_on, ramp_off, **kwargs):
        super(HalfSineWaveform, self).__init__(
            ramp_on=ramp_on, ramp_off=ramp_off, **kwargs
        )

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
                self.s_eDeriv(simulation, time, v, adjoint),
            )
        else:
            return (
                lambda v: self.s_mDeriv(simulation, time, v, adjoint),
                lambda v: self.s_eDeriv(simulation, time, v, adjoint),
            )

    def s_m(self, simulation, time):
        return Zero()

    def s_e(self, simulation, time):
        return Zero()

    def s_mDeriv(self, simulation, time, v=None, adjoint=False):
        return Zero()

    def s_eDeriv(self, simulation, time, v=None, adjoint=False):
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
                location=self.location,
                moment=self.moment,
            )
        return self._dipole.vector_potential(obsLoc, coordinates=coordinates)

    def _aSrc(self, simulation):
        coordinates = "cartesian"
        if simulation._formulation == "EB":
            gridX = simulation.mesh.gridEx
            gridY = simulation.mesh.gridEy
            gridZ = simulation.mesh.gridEz

        elif simulation._formulation == "HJ":
            gridX = simulation.mesh.gridFx
            gridY = simulation.mesh.gridFy
            gridZ = simulation.mesh.gridFz

        if simulation.mesh._meshType == "CYL":
            coordinates = "cylindrical"
            if simulation.mesh.isSymmetric:
                return self._srcFct(gridY)[:, 1]

        ax = self._srcFct(gridX, coordinates)[:, 0]
        ay = self._srcFct(gridY, coordinates)[:, 1]
        az = self._srcFct(gridZ, coordinates)[:, 2]
        a = np.concatenate((ax, ay, az))

        return a

    def _getAmagnetostatic(self, simulation):
        if simulation._formulation == "EB":
            return (
                simulation.mesh.faceDiv * simulation.MfMuiI * simulation.mesh.faceDiv.T
            )
        else:
            raise NotImplementedError(
                "Solving the magnetostatic simulationlem for the initial fields "
                "when a permeable model is considered has not yet been "
                "implemented for the HJ formulation. "
                "See: https://github.com/simpeg/simpeg/issues/680"
            )

    def _rhs_magnetostatic(self, simulation):
        if getattr(self, "_hp", None) is None:
            if simulation._formulation == "EB":
                bp = simulation.mesh.edgeCurl * self._aSrc(simulation)
                self._MfMuip = simulation.mesh.getFaceInnerProduct(1.0 / self.mu)
                self._MfMuipI = simulation.mesh.getFaceInnerProduct(
                    1.0 / self.mu, invMat=True
                )
                self._hp = self._MfMuip * bp
            else:
                raise NotImplementedError(
                    "Solving the magnetostatic simulationlem for the initial fields "
                    "when a permeable model is considered has not yet been "
                    "implemented for the HJ formulation. "
                    "See: https://github.com/simpeg/simpeg/issues/680"
                )

        if simulation._formulation == "EB":
            return -simulation.mesh.faceDiv * (
                (simulation.MfMuiI - self._MfMuipI) * self._hp
            )
        else:
            raise NotImplementedError(
                "Solving the magnetostatic simulationlem for the initial fields "
                "when a permeable model is considered has not yet been "
                "implemented for the HJ formulation. "
                "See: https://github.com/simpeg/simpeg/issues/680"
            )

    def _phiSrc(self, simulation):
        Ainv = simulation.solver(
            self._getAmagnetostatic(simulation)
        )  # todo: store these
        rhs = self._rhs_magnetostatic(simulation)
        Ainv.clean()
        return Ainv * rhs

    def _bSrc(self, simulation):
        if simulation._formulation == "EB":
            C = simulation.mesh.edgeCurl

        elif simulation._formulation == "HJ":
            C = simulation.mesh.edgeCurl.T

        return C * self._aSrc(simulation)

    def bInitial(self, simulation):

        if self.waveform.has_initial_fields is False:
            return Zero()

        if np.all(simulation.mu == self.mu):
            return self._bSrc(simulation)

        else:
            if simulation._formulation == "EB":
                hs = simulation.mesh.faceDiv.T * self._phiSrc(simulation)
                ht = self._hp + hs
                return simulation.MfMuiI * ht
            else:
                raise NotImplementedError

    def hInitial(self, simulation):

        if self.waveform.has_initial_fields is False:
            return Zero()
        # if simulation._formulation == 'EB':
        #     return simulation.MfMui * self.bInitial(simulation)
        # elif simulation._formulation == 'HJ':
        #     return simulation.MeMuI * self.bInitial(simulation)
        return 1.0 / self.mu * self.bInitial(simulation)

    def s_m(self, simulation, time):
        if self.waveform.has_initial_fields is False:
            return Zero()
        return Zero()

    def s_e(self, simulation, time):
        C = simulation.mesh.edgeCurl
        b = self._bSrc(simulation)

        if simulation._formulation == "EB":
            MfMui = simulation.mesh.getFaceInnerProduct(1.0 / self.mu)

            if (
                self.waveform.has_initial_fields is True
                and time < simulation.time_steps[1]
            ):
                if simulation._fieldType == "b":
                    return Zero()
                elif simulation._fieldType == "e":
                    # Compute s_e from vector potential
                    return C.T * (MfMui * b)
            else:
                return C.T * (MfMui * b) * self.waveform.eval(time)

        elif simulation._formulation == "HJ":

            h = 1.0 / self.mu * b

            if (
                self.waveform.has_initial_fields is True
                and time < simulation.time_steps[1]
            ):
                if simulation._fieldType == "h":
                    return Zero()
                elif simulation._fieldType == "j":
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
        return np.pi * self.radius**2 * self.current * self.N

    def _srcFct(self, obsLoc, coordinates="cartesian"):
        # return MagneticLoopVectorPotential(
        #     self.location, obsLoc, component, mu=self.mu, radius=self.radius
        # )

        if getattr(self, "_loop", None) is None:
            self._loop = CircularLoopWholeSpace(
                mu=self.mu,
                location=self.location,
                orientation=self.orientation,
                radius=self.radius,
                current=self.current,
            )
        return self._loop.vector_potential(obsLoc, coordinates)


class LineCurrent(BaseTDEMSrc):
    """
    Line current source.

    :param list receiver_list: receiver list
    :param bool integrate: Integrate the source term (multiply by Me) [False]
    """

    location = properties.Array("location of the source", shape=("*", 3))
    current = properties.Float("current in the line", default=1.0)

    def __init__(self, receiver_list=None, **kwargs):
        self.integrate = False
        kwargs.pop("srcType", None)  # TODO: generalize this to loop sources
        super(LineCurrent, self).__init__(receiver_list, srcType="galvanic", **kwargs)

    def Mejs(self, simulation):
        if getattr(self, "_Mejs", None) is None:
            self._Mejs = segmented_line_current_source_term(
                simulation.mesh, self.location
            )
        return self.current * self._Mejs

    def Mfjs(self, simulation):
        if getattr(self, "_Mfjs", None) is None:
            self._Mfjs = line_through_faces(
                simulation.mesh, self.location, normalize_by_area=True
            )
        return self.current * self._Mfjs

    def getRHSdc(self, simulation):
        if simulation._formulation == "EB":
            Grad = simulation.mesh.nodalGrad
            return Grad.T * self.Mejs(simulation)
        elif simulation._formulation == "HJ":
            Div = sdiag(simulation.mesh.vol) * simulation.mesh.faceDiv
            return Div * self.Mfjs(simulation)

    def phiInitial(self, simulation):
        if self.waveform.has_initial_fields:
            RHSdc = self.getRHSdc(simulation)
            phi = simulation.Adcinv * RHSdc
            return phi
        else:
            return Zero()

    def _phiInitialDeriv(self, simulation, v, adjoint=False):
        if self.waveform.has_initial_fields:
            phi = self.phiInitial(simulation)

            if adjoint is True:
                return -1.0 * simulation.getAdcDeriv(
                    phi, simulation.Adcinv * v, adjoint=True
                )  # A is symmetric

            Adc_deriv = simulation.getAdcDeriv(phi, v)
            return -1.0 * (simulation.Adcinv * Adc_deriv)

        else:
            return Zero()

    def eInitial(self, simulation):
        if self.waveform.has_initial_fields:
            if simulation._formulation == "EB":
                phi = self.phiInitial(simulation)
                return -simulation.mesh.nodalGrad * phi
            else:
                raise NotImplementedError
        else:
            return Zero()

    def eInitialDeriv(self, simulation, v=None, adjoint=False, f=None):
        if self.waveform.has_initial_fields:
            edc = f[self, "e", 0]
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

    def jInitial(self, simulation):
        if self.waveform.has_initial_fields:
            if simulation._formulation == "HJ":
                phi = self.phiInitial(simulation)
                Div = sdiag(simulation.mesh.vol) * simulation.mesh.faceDiv
                return -simulation.MfRhoI * (Div.T * phi)
            else:
                raise NotImplementedError
        else:
            return Zero()

    def jInitialDeriv(self, simulation, v=None, adjoint=False, f=None):
        if self.waveform.has_initial_fields is False:
            return Zero()
        elif simulation._formulation != "HJ":
            raise NotImplementedError

        phi = self.phiInitial(simulation)
        Div = sdiag(simulation.mesh.vol) * simulation.mesh.faceDiv

        if adjoint is True:
            return -(
                simulation.MfRhoIDeriv(Div.T * phi, v=v, adjoint=True)
                + self._phiInitialDeriv(
                    simulation, Div * (simulation.MfRhoI.T * v), adjoint=True
                )
            )
        phiDeriv = self._phiInitialDeriv(simulation, v)
        return -(
            simulation.MfRhoIDeriv(Div.T * phi, v=v)
            + simulation.MfRhoI * (Div.T * phiDeriv)
        )

    def _getAmmr(self, simulation):
        if simulation._formulation != "HJ":
            raise NotImplementedError

        vol = simulation.mesh.vol
        Div = sdiag(vol) * simulation.mesh.faceDiv
        return (
            simulation.mesh.edgeCurl * simulation.MeMuI * simulation.mesh.edgeCurl.T
            - Div.T
            * sdiag(1.0 / vol * simulation.mui)
            * Div  # stabalizing term. See (Chen, Haber & Oldenburg 2002)
        )

    def _aInitial(self, simulation):
        A = self._getAmmr(simulation)
        Ainv = simulation.solver(A)  # todo: store this
        s_e = self.s_e(simulation, 0)
        rhs = s_e + self.jInitial(simulation)
        return Ainv * rhs

    def _aInitialDeriv(self, simulation, v, adjoint=False):
        A = self._getAmmr(simulation)
        Ainv = simulation.solver(A)  # todo: store this - move it to the simulation

        if adjoint is True:
            return self.jInitialDeriv(
                simulation, Ainv * v, adjoint=True
            )  # A is symmetric

        return Ainv * self.jInitialDeriv(simulation, v)

    def hInitial(self, simulation):
        if self.waveform.has_initial_fields is False:
            return Zero()
        elif simulation._formulation != "HJ":
            raise NotImplementedError

        if self.waveform.has_initial_fields is False:
            return Zero()

        b = self.bInitial(simulation)
        return simulation.MeMuI * b

    def hInitialDeriv(self, simulation, v, adjoint=False, f=None):
        if self.waveform.has_initial_fields is False:
            return Zero()
        elif simulation._formulation != "HJ":
            raise NotImplementedError

        if self.waveform.has_initial_fields is False:
            return Zero()

        if adjoint is True:
            return self.bInitialDeriv(simulation, simulation.MeMuI.T * v, adjoint=True)
        return simulation.MeMuI * self.bInitialDeriv(simulation, v)

    def bInitial(self, simulation):
        if self.waveform.has_initial_fields is False:
            return Zero()
        elif simulation._formulation != "HJ":
            raise NotImplementedError

        a = self._aInitial(simulation)
        return simulation.mesh.edgeCurl.T * a

    def bInitialDeriv(self, simulation, v, adjoint=False, f=None):
        if self.waveform.has_initial_fields is False:
            return Zero()
        elif simulation._formulation != "HJ":
            raise NotImplementedError

        if adjoint is True:
            return self._aInitialDeriv(
                simulation, simulation.mesh.edgeCurl * v, adjoint=True
            )
        return simulation.mesh.edgeCurl.T * self._aInitialDeriv(simulation, v)

    def s_m(self, simulation, time):
        return Zero()

    def s_e(self, simulation, time):
        if simulation._formulation == "EB":
            return self.Mejs(simulation) * self.waveform.eval(time)
        elif simulation._formulation == "HJ":
            return self.Mfjs(simulation) * self.waveform.eval(time)


# TODO: this should be generalized and plugged into getting the Line current
# on faces
class RawVec_Grounded(LineCurrent):

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
            self._Mfjs = self._s_e = s_e

    # def getRHSdc(self, simulation):
    #     return sdiag(simulation.mesh.vol) * simulation.mesh.faceDiv * self._s_e

    # def phiInitial(self, simulation):
    #     if self.waveform.has_initial_fields:
    #         RHSdc = self.getRHSdc(simulation)
    #         return simulation.Adcinv * RHSdc
    #     else:
    #         return Zero()

    # def _phiInitialDeriv(self, simulation, v, adjoint=False):
    #     if self.waveform.has_initial_fields:
    #         phi = self.phiInitial(simulation)

    #         if adjoint is True:
    #             return -1.0 * simulation.getAdcDeriv(
    #                 phi, simulation.Adcinv * v, adjoint=True
    #             )  # A is symmetric

    #         Adc_deriv = simulation.getAdcDeriv(phi, v)
    #         return -1.0 * (simulation.Adcinv * Adc_deriv)

    #     else:
    #         return Zero()

    # def jInitial(self, simulation):
    #     if simulation._fieldType not in ["j", "h"]:
    #         raise NotImplementedError

    #     if self.waveform.has_initial_fields is False:
    #         return Zero()

    #     phi = self.phiInitial(simulation)
    #     Div = sdiag(simulation.mesh.vol) * simulation.mesh.faceDiv
    #     return -simulation.MfRhoI * (Div.T * phi)

    # def jInitialDeriv(self, simulation, v, adjoint=False):
    #     if simulation._fieldType not in ["j", "h"]:
    #         raise NotImplementedError

    #     if self.waveform.has_initial_fields is False:
    #         return Zero()

    #     phi = self.phiInitial(simulation)
    #     Div = sdiag(simulation.mesh.vol) * simulation.mesh.faceDiv

    #     if adjoint is True:
    #         return -(
    #             simulation.MfRhoIDeriv(Div.T * phi, v=v, adjoint=True)
    #             + self._phiInitialDeriv(simulation, Div * (simulation.MfRhoI.T * v), adjoint=True)
    #         )
    #     phiDeriv = self._phiInitialDeriv(simulation, v)
    #     return -(simulation.MfRhoIDeriv(Div.T * phi, v=v) + simulation.MfRhoI * (Div.T * phiDeriv))

    # def _getAmmr(self, simulation):
    #     if simulation._fieldType not in ["j", "h"]:
    #         raise NotImplementedError

    #     vol = simulation.mesh.vol

    #     return (
    #         simulation.mesh.edgeCurl * simulation.MeMuI * simulation.mesh.edgeCurl.T
    #         - simulation.mesh.faceDiv.T
    #         * sdiag(1.0 / vol * simulation.mui)
    #         * simulation.mesh.faceDiv  # stabalizing term. See (Chen, Haber & Oldenburg 2002)
    #     )

    # def _aInitial(self, simulation):
    #     A = self._getAmmr(simulation)
    #     Ainv = simulation.solver(A)  # todo: store this
    #     s_e = self.s_e(simulation, 0)
    #     rhs = s_e - self.jInitial(simulation)
    #     return Ainv * rhs

    # def _aInitialDeriv(self, simulation, v, adjoint=False):
    #     A = self._getAmmr(simulation)
    #     Ainv = simulation.solver(A)  # todo: store this - move it to the simulationlem

    #     if adjoint is True:
    #         return -1 * (
    #             self.jInitialDeriv(simulation, Ainv * v, adjoint=True)
    #         )  # A is symmetric

    #     return -1 * (Ainv * self.jInitialDeriv(simulation, v))

    # def hInitial(self, simulation):
    #     if simulation._fieldType not in ["j", "h"]:
    #         raise NotImplementedError

    #     if self.waveform.has_initial_fields is False:
    #         return Zero()

    #     b = self.bInitial(simulation)
    #     return simulation.MeMuI * b

    # def hInitialDeriv(self, simulation, v, adjoint=False, f=None):
    #     if simulation._fieldType not in ["j", "h"]:
    #         raise NotImplementedError

    #     if self.waveform.has_initial_fields is False:
    #         return Zero()

    #     if adjoint is True:
    #         return self.bInitialDeriv(simulation, simulation.MeMuI.T * v, adjoint=True)
    #     return simulation.MeMuI * self.bInitialDeriv(simulation, v)

    # def bInitial(self, simulation):
    #     if simulation._fieldType not in ["j", "h"]:
    #         raise NotImplementedError

    #     if self.waveform.has_initial_fields is False:
    #         return Zero()

    #     a = self._aInitial(simulation)
    #     return simulation.mesh.edgeCurl.T * a

    # def bInitialDeriv(self, simulation, v, adjoint=False, f=None):
    #     if simulation._fieldType not in ["j", "h"]:
    #         raise NotImplementedError

    #     if self.waveform.has_initial_fields is False:
    #         return Zero()

    #     if adjoint is True:
    #         return self._aInitialDeriv(simulation, simulation.mesh.edgeCurl * v, adjoint=True)
    #     return simulation.mesh.edgeCurl.T * self._aInitialDeriv(simulation, v)

    # def s_e(self, simulation, time):
    #     # if simulation._fieldType == 'h':
    #     #     return simulation.Mf * self._s_e * self.waveform.eval(time)
    #     return self._s_e * self.waveform.eval(time)
