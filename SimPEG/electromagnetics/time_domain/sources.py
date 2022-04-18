import numpy as np
from scipy.constants import mu_0
# import properties
import warnings
from ...utils.code_utils import deprecate_property

from geoana.em.static import MagneticDipoleWholeSpace, CircularLoopWholeSpace

from ..base import BaseEMSrc
from ..utils import segmented_line_current_source_term, line_through_faces
from ...props import LocationVector
from ...utils import set_kwargs, sdiag, Zero

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
        set_kwargs(self, **kwargs)

    @property
    def has_initial_fields(self):
        """Whether the waveform has initial fields.

        If the current at the first time in the defined waveform is non-zero,
        there are static fields that must be computed by the simulations.
        `has_initial_fields` tells the simulation that initial fields must be
        computed for the corresponding source.

        Returns
        -------
        bool
            If ``True``, the waveform has initial fields. 
        """
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
        """Off-time

        Sets the start of the off-time for the waveform.

        Returns
        -------
        float
            The start of the off-time for the waveform
        """
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
        """Window of time within which the waveform is considered on

        Returns
        -------
        float
            Window of time within which the waveform is considered on
        """
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
        """Not implemented for base TDEM source"""
        raise NotImplementedError

    def evalDeriv(self, time):
        """Not implemented for base TDEM source"""
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
    off_time: float, default=0.
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
        """Evaluate step-off waveform at a given time

        Parameters
        ----------
        time : float
            A time

        Returns
        -------
        float
            Source current at the time provided
        """
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
        """Evaluate linear ramp-off waveform at a given time

        Parameters
        ----------
        time : float
            A time

        Returns
        -------
        float
            Source current at the time provided
        """
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
        set_kwargs(self, **kwargs)

    @property
    def waveform_function(self):
        """Function handle for a custom waveform

        Returns
        -------
        function
            A function that returns the source current for an input
            argument *time*.
        """
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
        """Evaluate custom waveform function at a given time

        Parameters
        ----------
        time : float
            A time

        Returns
        -------
        float
            Source current at the time provided
        """
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
        set_kwargs(self, **kwargs)

    @property
    def peak_time(self):
        """Peak time

        Returns
        -------
        float
            The peak time for the VTEM waveform
        """
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
        """Ramp on rate

        Parameter controlling how quickly the waveform ramps on (default is 3)

        Returns
        -------
        float
            Ramp on rate
        """
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
        """Evaluate VTEM waveform at a given time

        Parameters
        ----------
        time : float
            A time

        Returns
        -------
        float
            Source current at the time provided
        """
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
        set_kwargs(self, **kwargs)

    @property
    def ramp_on(self):
        """times over which the transmitter ramps on.

        Defines the start and end times over which the ramp-on occurs.

        Returns
        -------
        (2) np.ndarray
            A numpy array (t_start, t_final), where *t_start* defines the start of the ramp-on
            and *t_end* defines the time where the ramp-on is completed.
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
        """times over which the transmitter ramps off.

        Defines the start and end times over which the ramp-off occurs.

        Returns
        -------
        (2) np.ndarray
            A numpy array (t_start, t_final), where *t_start* defines the start of the ramp-off
            and *t_end* defines the time where the ramp-off is completed.
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
        """Evaluate the waveform at a given time

        Parameters
        ----------
        time : float
            A time

        Returns
        -------
        float
            Source current at the time provided
        """
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
        set_kwargs(self, **kwargs)

    @property
    def peak_time(self):
        """Peak time

        Returns
        -------
        float
            The peak time for the triangular waveform
        """
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
        """Evaluate the waveform at a given time

        Parameters
        ----------
        time : float
            A time

        Returns
        -------
        float
            Source current at the time provided
        """
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
        """Evaluate the waveform at a given time

        Parameters
        ----------
        time : float
            A time

        Returns
        -------
        float
            Source current at the time provided
        """
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
    """Base TDEM source class

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.frequency_domain.receivers.BaseRx
        A list of FDEM receivers
    location : (dim) np.ndarray
        Source locations
    waveform : BaseWaveform, default=StepOffWaveform
        A SimPEG waveform object
    source_type : str in {'inductive','galvanic'}
        Implement as an inductive or galvanic source
    """

    # # rxPair = Rx
    # waveform = properties.Instance(
    #     "A source waveform", BaseWaveform, default=StepOffWaveform()
    # )
    # srcType = properties.StringChoice(
    #     "is the source a galvanic of inductive source",
    #     choices=["inductive", "galvanic"],
    # )

    def __init__(self, receiver_list=None, location=None, waveform=StepOffWaveform(), srcType=None, **kwargs):

        super(BaseTDEMSrc, self).__init__(
            receiver_list=receiver_list, location=location, **kwargs
        )

        self.waveform = waveform
        if srcType is not None:
            self.srcType = srcType

    @property
    def waveform(self):
        """Current waveform for the source

        Returns
        -------
        BaseWaveform
            A SimPEG waveform
        """
        return self._waveform

    @waveform.setter
    def waveform(self, wave):

        if isinstance(wave, BaseWaveform):
            self._waveform = wave
        else:
            raise TypeError(f"Must be an instance of 'BaseWaveform', Got {type(wave)}")
    
    @property
    def srcType(self):
        """Implement at inductive or galvanic source

        Returns
        -------
        str
            Either 'inductive' or 'galvanic'
        """
        return self._srcType

    @srcType.setter
    def srcType(self, var):
        if isinstance(var, str):
            if var in ['inductive', 'galvanic']:
                self._srcType = var
            else:
                raise ValueError("srcType must be either 'inductive' or 'galvanic'.")
        else:
            raise TypeError(f"srcType must be str, got {type(var)}")

    def bInitial(self, simulation):
        """Return initial B-field (``Zero`` for ``BaseTDEMSrc`` class)

        Parameters
        ----------
        simulation : BaseTDEMSrc
            TDEM source

        Returns
        -------
        Zero
            Returns ``Zero`` for ``BaseTDEMSrc``
        """
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
        """Return magnetic and electric source terms at a given time

        Parameters
        ----------
        simulation : SimPEG.electromagnetics.base.BaseTDEMSimulation
            An instance of a time-domain electromagnetic simulation
        time : float
            The time at which you want to compute the source terms

        Returns
        -------
        tuple
            A tuple (s_m, s_e), where s_m is the discretized magnetic source term
            and s_e is the discretized electric course term.
        """
        s_m = self.s_m(simulation, time)
        s_e = self.s_e(simulation, time)
        return s_m, s_e

    def evalDeriv(self, simulation, time, v=None, adjoint=False):
        """Derivative of magnetic and electric source terms time a vector at a given time

        Parameters
        ----------
        simulation : SimPEG.electromagnetics.base.BaseTDEMSimulation
            An instance of a time-domain electromagnetic simulation
        time :
            The time at which you want to compute the derivative
        v : np.ndarray
            A vector
        adjoint : bool
            If ``True``, return the adjoint operation

        Returns
        -------
        tuple
            A tuple (s_mDeriv, s_eDerive). If `v` is not ``None``, the method returns
            the derivatives of the magnetic and electric sources times the vector `v`.
            If `v` is ``None``, the method returns the functions for multiplying the
            derivatives with a vector.
        """
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
    r"""
    Point magnetic dipole source calculated by taking the curl of a magnetic
    vector potential. By taking the discrete curl, we ensure that the magnetic
    flux density is divergence free (no magnetic monopoles!).

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.time_domain.receivers.BaseRx
        A list of TDEM receivers
    location : (dim) np.ndarray, default = np.r_[0., 0., 0.]
        Source location.
    moment : float
        Magnetic dipole moment amplitude
    mu : float
        Background magnetic permeability
    source_type : str in {'inductive','galvanic'}
        Implement as an inductive or galvanic source
    """

    # moment = properties.Float("dipole moment of the transmitter", default=1.0, min=0.0)
    # mu = properties.Float("permeability of the background", default=mu_0, min=0.0)
    # orientation = properties.Vector3(
    #     "orientation of the source", default="Z", length=1.0, required=True
    # )
    # location = LocationVector(
    #     "location of the source", default=np.r_[0.0, 0.0, 0.0], shape=(3,)
    # )
    # loc = deprecate_property(
    #     location, "loc", new_name="location", removal_version="0.16.0", error=True
    # )

    # def __init__(self, receiver_list=None, **kwargs):
    #     kwargs.pop("srcType", None)
    #     BaseTDEMSrc.__init__(
    #         self, receiver_list=receiver_list, srcType="inductive", **kwargs
    #     )

    def __init__(
        self,
        receiver_list=None,
        location=np.r_[0.0, 0.0, 0.0],
        moment=1.,
        orientation='z',
        mu=mu_0,
        srcType='inductive',
        **kwargs
    ):

        super(MagDipole, self).__init__(
            receiver_list=receiver_list, location=location, **kwargs
        )

        self.moment = moment
        self.orientation = orientation
        self.mu = mu
        self.srcType = srcType

    @property
    def location(self):
        """Location of the dipole

        Returns
        -------
        (3) numpy.ndarray of float
            xyz dipole location
        """
        return self._location

    @location.setter
    def location(self, vec):

        try:
            vec = np.atleast_1d(vec).astype(float).squeeze()
        except:
            raise TypeError(f"location must be array_like, got {type(vec)}")

        if len(vec) > 3:
            raise ValueError(
                f"location must be array_like with shape (3), got {len(vec)}"
            )

        self._location = vec

    @property
    def moment(self):
        """Amplitude of the dipole moment of the magnetic dipole (:math:`A/m^2`)

        Returns
        -------
        float
            Amplitude of the dipole moment of the magnetic dipole (:math:`A/m^2`)
        """
        return self._moment

    @moment.setter
    def moment(self, value):

        try:
            value = float(value)
        except:
            raise TypeError(f"moment must be a number, got {type(value)}")

        if value <= 0.0:
            raise ValueError("moment must be greater than 0")

        self._moment = value

    @property
    def orientation(self):
        """Orientation of the dipole as a normalized vector

        Returns
        -------
        (3) numpy.ndarray of float or str in {'x','y','z'}
            dipole orientation, normalized to unit magnitude
        """
        return self._orientation

    @orientation.setter
    def orientation(self, var):

        if isinstance(var, str):
            if var.lower() == 'x':
                var = np.r_[1., 0., 0.]
            elif var.lower() == 'y':
                var = np.r_[0., 1., 0.]
            elif var.lower() == 'z':
                var = np.r_[0., 0., 1.]
        else:
            try:
                var = np.atleast_1d(var).astype(float)
            except:
                raise TypeError(f"orientation must be str or array_like, got {type(var)}")

            if len(var) != 3:
                raise ValueError(
                    f"orientation must be array_like with shape (3,), got {len(var)}"
                )

        # Normalize the orientation
        var /= np.sqrt(np.sum(var**2))

        self._orientation = var

    @property
    def mu(self):
        """Magnetic permeability in H/m

        Returns
        -------
        float
            Magnetic permeability in H/m
        """
        return self._mu

    @mu.setter
    def mu(self, value):

        try:
            value = float(value)
        except:
            raise TypeError(f"mu must be a number, got {type(value)}")

        if value <= 0.0:
            raise ValueError("mu must be greater than 0")

        self._mu = value
    
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
        """Compute initial magnetic flux density.

        Note that we compute analytic vector potential and take numerical
        curl do it is divergence free on the mesh.

        Parameters
        ----------
        simulation : BaseTDEMSimulation
            A SimPEG TDEM simulation

        Returns
        -------
        numpy.ndarray
            Initial magnetic flux density
        """

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
        """Compute initial magnetic field.

        Note that we compute analytic vector potential and take numerical
        curl so that B is divergence-free before converting to H-field.

        Parameters
        ----------
        simulation : BaseTDEMSimulation
            A SimPEG TDEM simulation

        Returns
        -------
        numpy.ndarray
            Initial magnetic field
        """

        if self.waveform.has_initial_fields is False:
            return Zero()
        # if simulation._formulation == 'EB':
        #     return simulation.MfMui * self.bInitial(simulation)
        # elif simulation._formulation == 'HJ':
        #     return simulation.MeMuI * self.bInitial(simulation)
        return 1.0 / self.mu * self.bInitial(simulation)

    def s_m(self, simulation, time):
        """Magnetic source term (s_m) at a given time

        Parameters
        ----------
        simulation : BaseTDEMSimulation
            SimPEG TDEM simulation
        time : float
            Evaluation time

        Returns
        -------
        numpy.ndarray
            magnetic source term on mesh.
        """
        if self.waveform.has_initial_fields is False:
            return Zero()
        return Zero()

    def s_e(self, simulation, time):
        """Electric source term (s_e) at a given time

        Parameters
        ----------
        simulation : BaseTDEMSimulation
            SimPEG TDEM simulation
        time : float
            Evaluation time

        Returns
        -------
        numpy.ndarray
            Electric source term on mesh.
        """
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
    """
    Circular loop magnetic source calculated by taking the curl of a magnetic
    vector potential. By taking the discrete curl, we ensure that the magnetic
    flux density is divergence free (no magnetic monopoles!).

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.time_domain.receivers.BaseRx
        A list of TDEM receivers
    location : (dim) np.ndarray, default = np.r_[0., 0., 0.]
        Source location.
    orientation : str, default = 'z'
        Loop orientation. One of ('x', 'y', 'z')
    radius : float, default = 1.
        Loop radius
    current : float, default = 1.
        Source current
    mu : float
        Background magnetic permeability
    srcType : str, default = 'inductive'
        'inductive' to implement as inductive source and 'galvanic' to implement
        as galvanic source
    N : int, default = 1
        Number of turns in the loop
    """

    # def __init__(self, receiver_list=None, **kwargs):
    #     super(CircularLoop, self).__init__(receiver_list, **kwargs)

    def __init__(
        self,
        receiver_list=None,
        location=np.r_[0.0, 0.0, 0.0],
        orientation='z',
        radius=1.,
        current=1.,
        N=1,
        mu=mu_0,
        srcType='inductive',
        **kwargs
    ):

        if 'moment' in kwargs:
            kwargs.pop('moment')

        BaseTDEMSrc.__init__(
            self,
            receiver_list=receiver_list,
            location=location,
            **kwargs
        )

        self.orientation = orientation
        self.radius = radius
        self.current = current
        self.N = N
        self.mu = mu
        self.srcType = srcType

    # radius = properties.Float("radius of the loop source", default=1.0, min=0.0)

    @property
    def radius(self):
        """Loop radius

        Returns
        -------
        float
            Loop radius
        """
        return self._radius

    @radius.setter
    def radius(self, rad):
        try:
            rad = float(rad)
        except:
            raise TypeError(f"radius must be int or float, got {type(rad)}")

        if rad < 0.:
            raise TypeError("radius must be a positive value")

        self._radius = rad

    # current = properties.Float("current in the loop", default=1.0)

    @property
    def current(self):
        """Source current

        Returns
        -------
        float
            Source current
        """
        return self._current

    @current.setter
    def current(self, I):
        try:
            I = float(I)
        except:
            raise TypeError(f"current must be int or float, got {type(I)}")

        if np.abs(I) == 0.:
            raise TypeError("current must be non-zero.")

        self._current = I

    # N = properties.Float("number of turns in the loop", default=1.0)

    @property
    def moment(self):
        """Dipole moment of the loop.

        The dipole moment is given by :math:`NI\\pi r^2`

        Returns
        -------
        float
            Dipole moment of the loop
        """
        return np.pi * self.radius ** 2 * self.current * self.N

    @moment.setter
    def moment(self):
        warnings.warn(
            "Moment is not set as a property. I is the product"
            "of the loop radius and transmitter current"
        )
        pass

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
    """Line current source.

    Given the wire path provided by the (n_loc, 3) locations array,
    the cells intersected by the wire path are identified and integrated
    source terms are computed.

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.time_domain.receivers.BaseRx
        List of TDEM receivers
    locations : (n,3) numpy.ndarray
        Array defining the node locations for the wire path. For inductive sources,
        you must close the loop.
    """

    # location = properties.Array("location of the source", shape=("*", 3))
    # loc = deprecate_property(
    #     location, "loc", new_name="location", removal_version="0.16.0", error=True
    # )
    # current = properties.Float("current in the line", default=1.0)

    # def __init__(self, receiver_list=None, **kwargs):
    #     self.integrate = False
    #     kwargs.pop("srcType", None)  # TODO: generalize this to loop sources
    #     super(LineCurrent, self).__init__(receiver_list, srcType="galvanic", **kwargs)

    def __init__(
        self,
        receiver_list=None,
        location=None,
        current=1.,
        mu=mu_0,
        **kwargs
    ):

        BaseTDEMSrc.__init__(
            self,
            receiver_list=receiver_list,
            location=location,
            **kwargs
        )

        self.integrate = False
        self.current = current
        self.mu = mu


    @property
    def location(self):
        """Line current nodes locations

        Returns
        -------
        (n, 3) np.ndarray
            Line current node locations.
        """
        return self._location

    @location.setter
    def location(self, loc):
        try:
            loc = np.atleast_2d(loc).astype(float)
        except:
            raise TypeError(f"location must be (n, 3) array_like, got {type(loc)}")

        if loc.ndim != 2:
            raise TypeError(f"location must be (n, 3) array_like, got {type(loc)}")

        self._location = loc

    @property
    def current(self):
        """Source current

        Returns
        -------
        float
            Source current
        """
        return self._current

    @current.setter
    def current(self, I):
        try:
            I = float(I)
        except:
            raise TypeError(f"current must be int or float, got {type(I)}")

        if np.abs(I) == 0.:
            raise TypeError("current must be non-zero.")

        self._current = I
    

    def Mejs(self, simulation):
        """Integrated electrical source term on edges

        Parameters
        ----------
        simulation : SimPEG.electromagnetics.time_domain.simulation.BaseTDEMSimulation
            Base TDEM simulation

        Returns
        -------
        numpy.ndarray of length (mesh.nE)
            Contains the source term for all x, y, and z edges of the mesh.
        """
        if getattr(self, "_Mejs", None) is None:
            self._Mejs = segmented_line_current_source_term(
                simulation.mesh, self.location
            )
        return self.current * self._Mejs

    def Mfjs(self, simulation):
        """Integrated electrical source term on faces

        Parameters
        ----------
        simulation : SimPEG.electromagnetics.time_domain.simulation.BaseTDEMSimulation
            Base TDEM simulation

        Returns
        -------
        numpy.ndarray of length (mesh.nF)
            Contains the source term for all x, y, and z faces of the mesh.
        """
        if getattr(self, "_Mfjs", None) is None:
            self._Mfjs = line_through_faces(
                simulation.mesh, self.location, normalize_by_area=True
            )
        return self.current * self._Mfjs

    def getRHSdc(self, simulation):
        """Right-hand side for galvanic source term

        Parameters
        ----------
        simulation : SimPEG.electromagnetics.time_domain.simulation.BaseTDEMSimulation
            Base TDEM simulation

        Returns
        -------
        numpy.ndarray
            Right-hand side of galvanic source term. On edges for 'EB' formulation,
            and on faces for 'HJ' formulation.
        """
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
        """Compute initial electric field

        Parameters
        ----------
        simulation : BaseTDEMSimulation
            A SimPEG TDEM simulation

        Returns
        -------
        numpy.ndarray
            Initial electric field
        """
        if self.waveform.has_initial_fields:
            if simulation._formulation == "EB":
                phi = self.phiInitial(simulation)
                return -simulation.mesh.nodalGrad * phi
            else:
                raise NotImplementedError
        else:
            return Zero()

    def eInitialDeriv(self, simulation, v=None, adjoint=False, f=None):
        """Compute derivative of initial electric field times a vector

        Parameters
        ----------
        simulation : BaseTDEMSimulation
            A SimPEG TDEM simulation
        v : np.ndarray
            A vector
        adjoint : bool
            If ``True``, return the adjoint

        Returns
        -------
        numpy.ndarray
            Derivative of initial electric field times a vector
        """
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
        """Compute initial current density

        Parameters
        ----------
        simulation : BaseTDEMSimulation
            A SimPEG TDEM simulation

        Returns
        -------
        numpy.ndarray
            Initial current density
        """
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
        """Compute derivative of initial current density times a vector

        Parameters
        ----------
        simulation : BaseTDEMSimulation
            A SimPEG TDEM simulation
        v : np.ndarray
            A vector
        adjoint : bool
            If ``True``, return the adjoint

        Returns
        -------
        numpy.ndarray
            Derivative of initial current density times a vector
        """
        if simulation._formulation != "HJ":
            raise NotImplementedError

        if self.waveform.has_initial_fields is False:
            return Zero()

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

        return (
            simulation.mesh.edgeCurl * simulation.MeMuI * simulation.mesh.edgeCurl.T
            - simulation.mesh.faceDiv.T
            * sdiag(1.0 / vol * simulation.mui)
            * simulation.mesh.faceDiv  # stabalizing term. See (Chen, Haber & Oldenburg 2002)
        )

    def _aInitial(self, simulation):
        A = self._getAmmr(simulation)
        Ainv = simulation.solver(A)  # todo: store this
        s_e = self.s_e(simulation, 0)
        rhs = s_e - self.jInitial(simulation)
        return Ainv * rhs

    def _aInitialDeriv(self, simulation, v, adjoint=False):
        A = self._getAmmr(simulation)
        Ainv = simulation.solver(A)  # todo: store this - move it to the simulationlem

        if adjoint is True:
            return -1 * (
                self.jInitialDeriv(simulation, Ainv * v, adjoint=True)
            )  # A is symmetric

        return -1 * (Ainv * self.jInitialDeriv(simulation, v))

    def hInitial(self, simulation):
        """Compute initial magnetic field.

        Note that we compute analytic vector potential and take numerical
        curl so that B is divergence-free before converting to H-field.

        Parameters
        ----------
        simulation : BaseTDEMSimulation
            A SimPEG TDEM simulation

        Returns
        -------
        numpy.ndarray
            Initial magnetic field
        """
        if simulation._formulation != "HJ":
            raise NotImplementedError

        if self.waveform.has_initial_fields is False:
            return Zero()

        b = self.bInitial(simulation)
        return simulation.MeMuI * b

    def hInitialDeriv(self, simulation, v, adjoint=False, f=None):
        """Compute derivative of intitial magnetic field times a vector

        Parameters
        ----------
        simulation : BaseTDEMSimulation
            A SimPEG TDEM simulation
        v : np.ndarray
            A vector
        adjoint : bool
            If ``True``, return the adjoint

        Returns
        -------
        numpy.ndarray
            Derivative of initial magnetic field times a vector
        """
        if simulation._formulation != "HJ":
            raise NotImplementedError

        if self.waveform.has_initial_fields is False:
            return Zero()

        if adjoint is True:
            return self.bInitialDeriv(simulation, simulation.MeMuI.T * v, adjoint=True)
        return simulation.MeMuI * self.bInitialDeriv(simulation, v)

    def bInitial(self, simulation):
        """Compute initial magnetic flux density.

        Note that we compute analytic vector potential and take numerical
        curl do it is divergence free on the mesh.

        Parameters
        ----------
        simulation : BaseTDEMSimulation
            A SimPEG TDEM simulation

        Returns
        -------
        numpy.ndarray
            Initial magnetic flux density
        """
        if simulation._formulation != "HJ":
            raise NotImplementedError

        if self.waveform.has_initial_fields is False:
            return Zero()

        a = self._aInitial(simulation)
        return simulation.mesh.edgeCurl.T * a

    def bInitialDeriv(self, simulation, v, adjoint=False, f=None):
        """Compute derivative of intitial magnetic flux density times a vector

        Parameters
        ----------
        simulation : BaseTDEMSimulation
            A SimPEG tDEM simulation
        v : np.ndarray
            A vector
        adjoint : bool
            If ``True``, return the adjoint

        Returns
        -------
        numpy.ndarray
            Derivative of initial magnetic flux density times a vector
        """
        if simulation._formulation != "HJ":
            raise NotImplementedError

        if self.waveform.has_initial_fields is False:
            return Zero()

        if adjoint is True:
            return self._aInitialDeriv(
                simulation, simulation.mesh.edgeCurl * v, adjoint=True
            )
        return simulation.mesh.edgeCurl.T * self._aInitialDeriv(simulation, v)

    def s_m(self, simulation, time):
        return Zero()

    def s_e(self, simulation, time):
        """Electric source term (s_e) at the time provided

        Parameters
        ----------
        simulation : BaseTDEMSimulation
            SimPEG TDEM simulation
        time : float
            Time

        Returns
        -------
        numpy.ndarray
            electric source term on mesh.
        """
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

    # _s_e = properties.Array("source term", shape=("*",))

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
