import warnings

import numpy as np
from geoana.em.static import CircularLoopWholeSpace, MagneticDipoleWholeSpace
from scipy.constants import mu_0

from ...utils import Zero, sdiag
from ...utils.code_utils import (
    deprecate_property,
    validate_callable,
    validate_direction,
    validate_float,
    validate_integer,
    validate_location_property,
    validate_ndarray_with_shape,
    validate_string,
    validate_type,
)
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
    has_initial_fields : bool, default: ``False``
        If the transmitter has non-zero current prior to the start of the simulation
        (e.g. a step-off waveform), set `has_initial_fields` to True
    off_time : float, default: 0.0
        Time when the transmitter current is zero in units of seconds.
    epsilon : float, default: 1e-9
        Small time-constant for which the transmitter is assumed to still be on for
    """

    def __init__(self, has_initial_fields=False, off_time=0.0, epsilon=1e-9, **kwargs):
        if kwargs.pop("hasInitialFields", None):
            raise AttributeError(
                "hasInitialFields was removed in 0.17.0 use has_initial_fields"
            )
        if kwargs.pop("offTime", None):
            raise AttributeError("offTime was removed in 0.17.0 use off_time")
        if kwargs.pop("eps", None):
            raise AttributeError("eps was removed in 0.17.0 use epsilon")

        self.has_initial_fields = has_initial_fields
        self.off_time = off_time
        self.epsilon = epsilon
        super().__init__(**kwargs)

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
        self._has_initial_fields = validate_type("has_initial_fields", value, bool)

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
        self._off_time = validate_float("off_time", value)

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
        self._epsilon = validate_float("epsilon", value, min_val=0)

    def eval(self, time):  # noqa: A003
        """Evaluate current waveform at a given time

        Parameters
        ----------
        time : float
            A time

        Returns
        -------
        float
            Source current at the time provided
        """
        raise NotImplementedError

    def eval_deriv(self, time):
        """Evaluate the time derivative of the linear ramp-off waveform at given times

        Parameters
        ----------
        time : float or numpy.ndarray
            Time(s) to evaluate the waveform derivative at.

        Returns
        -------
        float or numpy.ndarray
            Source current at the time provided
        """
        raise NotImplementedError  # needed for E-formulation


class StepOffWaveform(BaseWaveform):
    """
    A heavy-side step function waveform. This is the default for time-domain EM simulations.

    Parameters
    ----------
    off_time : float, default: 0.0
        time at which the transmitter is turned off in units of seconds (default is 0s)

    Examples
    --------
    The default off-time for the step-off waveform is 0s. In the example below, we set it to
    1e-5 s (0.01 ms) to illustrate it in a plot

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from simpeg.electromagnetics import time_domain as tdem

    >>> times = np.linspace(0, 1e-4, 1000)
    >>> waveform = tdem.sources.StepOffWaveform(off_time=1e-5)
    >>> plt.plot(times, [waveform.eval(t) for t in times])
    >>> plt.show()

    """

    def __init__(self, off_time=0.0, **kwargs):
        super().__init__(off_time=off_time, has_initial_fields=True, **kwargs)

    def eval(self, time):  # noqa: A003
        if (abs(time - 0.0) < self.epsilon) or ((time - self.off_time) < self.epsilon):
            return 1.0
        else:
            return 0.0


class RampOffWaveform(BaseWaveform):
    """
    A waveform with a linear ramp-off.

    Parameters
    ----------
    off_time : float, default: 0.0
        time at which the transmitter is turned off in units of seconds

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from simpeg.electromagnetics import time_domain as tdem

    >>> times = np.linspace(0, 1e-4, 1000)
    >>> waveform = tdem.sources.RampOffWaveform(off_time=1e-5)
    >>> plt.plot(times, [waveform.eval(t) for t in times])
    >>> plt.show()

    """

    def __init__(self, off_time=0.0, **kwargs):
        super().__init__(off_time=off_time, has_initial_fields=True, **kwargs)

    def eval(self, time):  # noqa: A003
        if abs(time - 0.0) < self.epsilon:
            return 1.0
        elif time < self.off_time:
            return -1.0 / self.off_time * (time - self.off_time)
        else:
            return 0.0

    def eval_deriv(self, time):
        t = np.asarray(time, dtype=float)
        out = np.zeros_like(t)

        if self.off_time > 0:
            out[(t < self.off_time) & (t >= self.epsilon)] = -1.0 / self.off_time

        if out.ndim == 0:
            out = out.item()
        return out

    @property
    def time_nodes(self):
        return np.r_[0.0, self.off_time]


class RawWaveform(BaseWaveform):
    """
    A waveform you can define. You need to provide a `waveform_function` that returns
    the waveform evaluated at a given time. This can be used, for example if you would
    like to interpolate between points specified in a waveform file.

    Parameters
    ----------
    off_time : float, default: 0.0
        time at which the transmitter is turned off in units of seconds (default is 0s)

    waveform_function: function

    Examples
    --------

    In this example, we define a saw-tooth waveform

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from simpeg.electromagnetics import time_domain as tdem

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
        if waveform_function is not None:
            self.waveform_function = waveform_function
        super().__init__(off_time=off_time, **kwargs)

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
        self._waveform_function = validate_callable("waveform_function", value)

    def eval(self, time):  # noqa: A003
        return self.waveform_function(time)


class VTEMWaveform(BaseWaveform):
    """
    A VTEM style waveform

    Parameters
    ----------
    off_time : float, default: 4.2e-3
        time at which the transmitter is turned off in units of seconds
    peak_time : float, default: 2.73e-3
        the peak time for the waveform
    ramp_on_rate : float, default: 3.0
        parameter controlling how quickly the waveform ramps on

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from simpeg.electromagnetics import time_domain as tdem

    >>> times = np.linspace(0, 1e-2, 1000)
    >>> waveform = tdem.sources.VTEMWaveform()
    >>> plt.plot(times, [waveform.eval(t) for t in times])
    >>> plt.show()

    """

    def __init__(self, off_time=4.2e-3, peak_time=2.73e-3, ramp_on_rate=3.0, **kwargs):
        if kwargs.pop("peakTime", None):
            raise AttributeError("peakTime was removed in 0.17.0, use peak_time")
        if kwargs.pop("a", None):
            raise AttributeError("a was removed in 0.17.0, use ramp_on_rate")
        super().__init__(has_initial_fields=False, off_time=off_time, **kwargs)
        self.ramp_on_rate = ramp_on_rate
        self.peak_time = peak_time

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
        value = validate_float("peak_time", value, max_val=self.off_time)
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
        value = validate_float("ramp_on_rate", value, min_val=0.0, inclusive_min=False)
        self._ramp_on_rate = value

    def eval(self, time):  # noqa: A003
        if time <= self.peak_time:
            return (1.0 - np.exp(-self.ramp_on_rate * time / self.peak_time)) / (
                1.0 - np.exp(-self.ramp_on_rate)
            )
        elif (time < self.off_time) and (time > self.peak_time):
            return -1.0 / (self.off_time - self.peak_time) * (time - self.off_time)
        else:
            return 0.0

    def eval_deriv(self, time):
        t = np.asarray(time, dtype=float)
        out = np.zeros_like(t)

        p_1 = (t <= self.peak_time) & (t >= 0.0)
        out[p_1] = (
            self.ramp_on_rate
            / self.peak_time
            * np.exp(-self.ramp_on_rate * t[p_1] / self.peak_time)
            / (1.0 - np.exp(-self.ramp_on_rate))
        )

        p_2 = (t > self.peak_time) & (t < self.off_time)
        out[p_2] = -1.0 / (self.off_time - self.peak_time)

        if out.ndim == 0:
            out = out.item()
        return out

    @property
    def time_nodes(self):
        return np.r_[0, self.peak_time, self.off_time]


class TrapezoidWaveform(BaseWaveform):
    """
    A waveform that has a linear ramp-on and a linear ramp-off.

    Parameters
    ----------
    ramp_on : (2) array_like of float
        time when the linear ramp_on starts and stops
    ramp_off : (2) array_like of float
        time when of the ramp_off starts and stops
    off_time : float
        time when the transmitter_current returns to zero

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from simpeg.electromagnetics import time_domain as tdem

    >>> times = np.linspace(0, 1e-2, 1000)
    >>> waveform = tdem.sources.TrapezoidWaveform(ramp_on=[0.0, 2e-3], ramp_off=[4e-3, 6e-3])
    >>> plt.plot(times, [waveform.eval(t) for t in times])
    >>> plt.show()

    """

    def __init__(self, ramp_on, ramp_off, off_time=None, **kwargs):
        super().__init__(has_initial_fields=False, **kwargs)
        self.ramp_on = ramp_on
        self.ramp_off = ramp_off
        self.off_time = off_time if off_time is not None else self.ramp_off[-1]

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
        self._ramp_on = validate_ndarray_with_shape(
            "ramp_on", value, shape=(2,), dtype=float
        )

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
        self._ramp_off = validate_ndarray_with_shape(
            "ramp_off", value, shape=(2,), dtype=float
        )

    def eval(self, time):  # noqa: A003
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

    def eval_deriv(self, time):
        t = np.asarray(time, dtype=float)
        out = np.zeros_like(t)

        p_1 = (t >= self.ramp_on[0]) & (t <= self.ramp_on[1])
        out[p_1] = 1.0 / (self.ramp_on[1] - self.ramp_on[0])

        p_2 = (t >= self.ramp_off[0]) & (t <= self.ramp_off[1])
        out[p_2] = -1.0 / (self.ramp_off[1] - self.ramp_off[0])

        if out.ndim == 0:
            out = out.item()
        return out

    @property
    def time_nodes(self):
        return np.unique(np.r_[self.ramp_on, self.ramp_off])


class TriangularWaveform(TrapezoidWaveform):
    """
    TriangularWaveform is a special case of TrapezoidWaveform where there's no pleateau

    Parameters
    ----------
    off_time : float
        time when the transmitter current returns to zero
    peak_time : float
        time when the transmitter waveform is at a peak

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from simpeg.electromagnetics import time_domain as tdem

    >>> times = np.linspace(0, 1e-2, 1000)
    >>> waveform = tdem.sources.TriangularWaveform(start_time=1E-3, off_time=6e-3, peak_time=3e-3)
    >>> plt.plot(times, [waveform.eval(t) for t in times])
    >>> plt.show()

    """

    def __init__(self, start_time, off_time, peak_time, **kwargs):
        if kwargs.get("startTime", None):
            AttributeError(
                "startTime will be deprecated in 0.17.0. Please update your code to use start_time instead",
            )
        if kwargs.get("peak_time", None):
            AttributeError(
                "peak_time will be deprecated in 0.17.0. Please update your code to use peak_time instead",
            )
        if kwargs.get("offTime", None):
            AttributeError(
                "offTime will be deprecated in 0.17.0. Please update your code to use off_time instead",
            )

        ramp_on = np.r_[start_time, peak_time]
        ramp_off = np.r_[peak_time, off_time]

        super().__init__(
            off_time=off_time,
            ramp_on=ramp_on,
            ramp_off=ramp_off,
            **kwargs,
        )
        self.peak_time = peak_time

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
        value = validate_float("peak_time", value, max_val=self.off_time)
        self._peak_time = value
        self._ramp_on = np.r_[self._ramp_on[0], value]
        self._ramp_off = np.r_[value, self._ramp_off[1]]


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
    >>> from simpeg.electromagnetics import time_domain as tdem

    >>> times = np.linspace(0, 1e-2, 1000)
    >>> waveform = tdem.sources.QuarterSineRampOnWaveform(ramp_on=(0, 2e-3), ramp_off=(3e-3, 3.5e-3))
    >>> plt.plot(times, [waveform.eval(t) for t in times])
    >>> plt.show()
    """

    def __init__(self, ramp_on, ramp_off, **kwargs):
        super().__init__(ramp_on=ramp_on, ramp_off=ramp_off, **kwargs)

    def eval(self, time):  # noqa: A003
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

    def eval_deriv(self, time):
        t = np.asarray(time, dtype=float)
        out = np.zeros_like(t)

        p_1 = (t >= self.ramp_on[0]) & (t < self.ramp_on[1])
        out[p_1] = (
            np.pi
            / 2
            / (self.ramp_on[1] - self.ramp_on[0])
            * np.cos(
                np.pi
                / 2
                * (t[p_1] - self.ramp_on[0])
                / (self.ramp_on[1] - self.ramp_on[0])
            )
        )

        p_2 = (t >= self.ramp_off[0]) & (t < self.ramp_off[1]) & (~p_1)
        out[p_2] = -1.0 / (self.ramp_off[1] - self.ramp_off[0])

        if out.ndim == 0:
            out = out.item()
        return out


class HalfSineWaveform(TrapezoidWaveform):
    """
    A waveform that has a quarter-sine ramp-on and a quarter-cosine ramp-off.

    When the end of ramp-on and start of ramp off are on the same spot, it looks
    like a half sine wave.

    Parameters
    ----------
    ramp_on: tuple
        times during which there is a quarter-sine ramp-on

    ramp_off: tuple
        times between which there is a quarter-cosine ramp-off.
    """

    def __init__(self, ramp_on, ramp_off, **kwargs):
        super().__init__(ramp_on=ramp_on, ramp_off=ramp_off, **kwargs)

    def eval(self, time):  # noqa: A003
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

    def eval_deriv(self, time):
        t = np.asarray(time, dtype=float)
        out = np.zeros_like(t)

        p_1 = (t >= self.ramp_on[0]) & (t < self.ramp_on[1])
        out[p_1] = (
            np.pi
            / 2
            / (self.ramp_on[1] - self.ramp_on[0])
            * np.cos(
                np.pi
                / 2
                * (t[p_1] - self.ramp_on[0])
                / (self.ramp_on[1] - self.ramp_on[0])
            )
        )

        p_2 = (t >= self.ramp_off[0]) & (t < self.ramp_off[1]) & (~p_1)
        out[p_2] = (
            -np.pi
            / 2
            / (self.ramp_off[1] - self.ramp_off[0])
            * np.sin(
                np.pi
                / 2
                * (t[p_2] - self.ramp_off[0])
                / (self.ramp_off[1] - self.ramp_off[0])
            )
        )

        if out.ndim == 0:
            out = out.item()
        return out


class PiecewiseLinearWaveform(BaseWaveform):
    """
    A waveform defined by a piecewise linear description of current.

    Parameters
    ----------
    times : array_like
        The time where current is defined.
    currents : array_like
        The values of the current.
    """

    def __init__(self, times, currents, **kwargs):
        times = validate_ndarray_with_shape("times", times, shape=("*",), dtype=float)
        currents = validate_ndarray_with_shape(
            "currents", currents, shape=("*",), dtype=float
        )
        if len(times) != len(currents):
            raise ValueError("time array and current array must be the same length")

        # ensure it is a sorted list...
        ind_sort = np.argsort(times)
        self.times = times[ind_sort]
        self.currents = currents[ind_sort]

        super().__init__(has_initial_fields=(self.currents[0] != 0.0), **kwargs)

    @property
    def times(self):
        """The times of defined current points of a piecewise linear waveform

        Returns
        -------
        numpy.ndarray of float

        """
        return self._times

    @times.setter
    def times(self, val):
        self._times = validate_ndarray_with_shape(
            "times", val, shape=("*",), dtype=float
        )

    @property
    def currents(self):
        """Current values for the piecewise linear waveform

        Returns
        -------
        numpy.ndarray of float
        """
        return self._currents

    @currents.setter
    def currents(self, val):
        self._currents = validate_ndarray_with_shape(
            "currents", val, shape=("*",), dtype=float
        )

    def eval(self, time):  # noqa: A003
        times = self.times
        currents = self.currents
        if time <= times[0]:
            return currents[0]
        elif time >= times[-1]:
            return currents[-1]
        else:
            i = np.searchsorted(times, time)
            return (currents[i] - currents[i - 1]) * (time - times[i - 1]) / (
                times[i] - times[i - 1]
            ) + currents[i - 1]

    def eval_deriv(self, time):
        t = np.asarray(time, dtype=float)
        out = np.zeros_like(t)

        times = self.times
        currents = self.currents
        p_1 = (t > times[0]) & (t < times[-1])

        i = np.searchsorted(times, t[p_1])

        out[p_1] = (currents[i] - currents[i - 1]) / (times[i] - times[i - 1])

        if out.ndim == 0:
            out = out.item()
        return out

    @property
    def time_nodes(self):
        return self.times


class ExponentialWaveform(BaseWaveform):
    """
    An exponential ramp-on and linear ramp-off waveform

    Parameters
    ----------
    start_time : float, default: -1e-2
        time at which the transmitter is turned on in units of seconds
    peak_time : float, default: -1e-3
        the peak time for the waveform in units of seconds
    ramp_on_tau : float, default: 1e-3
        time constant tau in units of seconds controlling how quickly the waveform ramps on (formula: 1-e^(-t/tau))

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from simpeg.electromagnetics import time_domain as tdem

    >>> times = np.linspace(-1e-2, 1e-2, 1000)
    >>> waveform = tdem.sources.ExponentialWaveform()
    >>> plt.plot(times, [waveform.eval(t) for t in times])
    >>> plt.show()

    """

    def __init__(
        self,
        start_time=-1e-2,
        peak_time=-1e-3,
        ramp_on_tau=1e-3,
        off_time=0.0,
        **kwargs,
    ):
        super().__init__(hasInitialFields=False, off_time=off_time, **kwargs)
        self.start_time = start_time
        self.peak_time = peak_time
        self.ramp_on_tau = ramp_on_tau

    @property
    def start_time(self):
        """Start time

        Returns
        -------
        float
            The start time for the Exponential waveform in units of seconds
        """
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        value = validate_float(
            "start_time", value, max_val=self.off_time, inclusive_max=False
        )
        self._start_time = value

    @property
    def peak_time(self):
        """Peak time

        Returns
        -------
        float
            The peak time for the Exponential waveform in units of seconds
        """
        return self._peak_time

    @peak_time.setter
    def peak_time(self, value):
        value = validate_float(
            "peak_time",
            value,
            min_val=self.start_time,
            max_val=self.off_time,
            inclusive_min=False,
        )
        self._peak_time = value

    @property
    def ramp_on_tau(self):
        """Ramp on rate control parameter.

        Time constant tau controlling how quickly the waveform ramps on in units of seconds

        Returns
        -------
        float
            Ramp on time constant tau
        """
        return self._ramp_on_tau

    @ramp_on_tau.setter
    def ramp_on_tau(self, value):
        value = validate_float("ramp_on_tau", value, min_val=0.0, inclusive_min=False)
        self._ramp_on_tau = value

    def eval(self, time):  # noqa: A003
        if time <= self.start_time:
            return 0.0
        elif time <= self.peak_time:
            # normalize to unity at the peak time
            return (1.0 - np.exp(-(time - self.start_time) / self.ramp_on_tau)) / (
                1.0 - np.exp(-(self.peak_time - self.start_time) / self.ramp_on_tau)
            )
        elif (time < self.off_time) and (time > self.peak_time):
            return -1.0 / (self.off_time - self.peak_time) * (time - self.off_time)
        else:
            return 0.0

    def eval_deriv(self, time):
        t = np.asarray(time, dtype=float)
        out = np.zeros_like(t)

        p_1 = (t <= self.peak_time) & (t >= self.start_time)
        out[p_1] = np.exp(-(t[p_1] - self.start_time) / self.ramp_on_tau) / (
            self.ramp_on_tau
            * (1.0 - np.exp(-(self.peak_time - self.start_time) / self.ramp_on_tau))
        )

        p_2 = (t > self.peak_time) & (t < self.off_time)
        out[p_2] = -1.0 / (self.off_time - self.peak_time)

        if out.ndim == 0:
            out = out.item()
        return out

    @property
    def time_nodes(self):
        return np.r_[self.start_time, self.peak_time, self.off_time]


###############################################################################
#                                                                             #
#                                    Sources                                  #
#                                                                             #
###############################################################################


class BaseTDEMSrc(BaseEMSrc):
    """Base TDEM source class

    Parameters
    ----------
    receiver_list : list of simpeg.electromagnetics.frequency_domain.receivers.BaseRx
        A list of FDEM receivers
    location : (dim) numpy.ndarray
        Source locations
    waveform : BaseWaveform, default=StepOffWaveform
        A SimPEG waveform object
    source_type : {'inductive','galvanic'}
        Implement as an inductive or galvanic source
    """

    def __init__(
        self,
        receiver_list=None,
        location=None,
        waveform=None,
        srcType=None,
        **kwargs,
    ):
        if waveform is None:
            waveform = StepOffWaveform()
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
        self._waveform = validate_type("waveform", wave, BaseWaveform, cast=False)

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
        self._srcType = validate_string("srcType", var, ["inductive", "galvanic"])

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
        """Returns :class:`Zero` for ``BaseTDEMSrc``"""
        return Zero()

    def eInitial(self, simulation):
        """Returns :class:`Zero` for ``BaseTDEMSrc``"""
        return Zero()

    def eInitialDeriv(self, simulation, v=None, adjoint=False, f=None):
        """Returns :class:`Zero` for ``BaseTDEMSrc``"""
        return Zero()

    def hInitial(self, simulation):
        """Returns :class:`Zero` for ``BaseTDEMSrc``"""
        return Zero()

    def hInitialDeriv(self, simulation, v=None, adjoint=False, f=None):
        """Returns :class:`Zero` for ``BaseTDEMSrc``"""
        return Zero()

    def jInitial(self, simulation):
        """Returns :class:`Zero` for ``BaseTDEMSrc``"""
        return Zero()

    def jInitialDeriv(self, simulation, v=None, adjoint=False, f=None):
        """Returns :class:`Zero` for ``BaseTDEMSrc``"""
        return Zero()

    def eval(self, simulation, time):  # noqa: A003
        """Return magnetic and electric source terms at a given time

        Parameters
        ----------
        simulation : simpeg.electromagnetics.base.BaseTDEMSimulation
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
        simulation : simpeg.electromagnetics.base.BaseTDEMSimulation
            An instance of a time-domain electromagnetic simulation
        time :
            The time at which you want to compute the derivative
        v : numpy.ndarray
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
        """Returns :class:`Zero` for ``BaseTDEMSrc``"""
        return Zero()

    def s_e(self, simulation, time):
        """Returns :class:`Zero` for ``BaseTDEMSrc``"""
        return Zero()

    def s_mDeriv(self, simulation, time, v=None, adjoint=False):
        """Returns :class:`Zero` for ``BaseTDEMSrc``"""
        return Zero()

    def s_eDeriv(self, simulation, time, v=None, adjoint=False):
        """Returns :class:`Zero` for ``BaseTDEMSrc``"""
        return Zero()


class MagDipole(BaseTDEMSrc):
    r"""
    Point magnetic dipole source calculated by taking the curl of a magnetic
    vector potential. By taking the discrete curl, we ensure that the magnetic
    flux density is divergence free (no magnetic monopoles!).

    Parameters
    ----------
    receiver_list : list of simpeg.electromagnetics.time_domain.receivers.BaseRx
        A list of TDEM receivers
    location : (dim) numpy.ndarray, default = np.r_[0., 0., 0.]
        Source location.
    moment : float
        Magnetic dipole moment amplitude
    orientation : {"z", "x", "y"} or (3) numpy.ndarray
        Orientation of the magnetic dipole.
    mu : float
        Background magnetic permeability
    source_type : {'inductive', 'galvanic'}
        Implement as an inductive or galvanic source
    """

    def __init__(
        self,
        receiver_list=None,
        location=None,
        moment=1.0,
        orientation="z",
        mu=mu_0,
        srcType="inductive",
        **kwargs,
    ):
        if location is None:
            location = np.r_[0.0, 0.0, 0.0]

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
        self._location = validate_location_property("location", vec, dim=3)

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
        value = validate_float("moment", value, min_val=0, inclusive_min=False)
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
        self._orientation = validate_direction("orientation", var, dim=3)

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
        value = validate_float("mu", value, min_val=mu_0)
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
            if simulation.mesh.is_symmetric:
                return self._srcFct(gridY)[:, 1]

        ax = self._srcFct(gridX, coordinates)[:, 0]
        ay = self._srcFct(gridY, coordinates)[:, 1]
        az = self._srcFct(gridZ, coordinates)[:, 2]
        a = np.concatenate((ax, ay, az))

        return a

    def _getAmagnetostatic(self, simulation):
        if simulation._formulation == "EB":
            return (
                simulation.mesh.face_divergence
                * simulation.MfMuiI
                * simulation.mesh.face_divergence.T.tocsr()
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
                bp = simulation.mesh.edge_curl * self._aSrc(simulation)
                self._MfMuip = simulation.mesh.get_face_inner_product(1.0 / self.mu)
                self._MfMuipI = simulation.mesh.get_face_inner_product(
                    1.0 / self.mu, invert_matrix=True
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
            return -simulation.mesh.face_divergence * (
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
            C = simulation.mesh.edge_curl

        elif simulation._formulation == "HJ":
            C = simulation.mesh.edge_curl.T

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
                hs = simulation.mesh.face_divergence.T * self._phiSrc(simulation)
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
        C = simulation.mesh.edge_curl
        b = self._bSrc(simulation)

        if simulation._formulation == "EB":
            MfMui = simulation.mesh.get_face_inner_product(1.0 / self.mu)

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
    receiver_list : list of simpeg.electromagnetics.time_domain.receivers.BaseRx
        A list of TDEM receivers
    location : (dim) np.ndarray, default = np.r_[0., 0., 0.]
        Source location.
    orientation : {'z', 'x', 'y'} or (3) numpy.ndarray
        Loop orientation.
    radius : float, default = 1.
        Loop radius
    current : float, default = 1.
        Source current
    mu : float
        Background magnetic permeability
    srcType : {'inductive', "galvanic"}
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
        location=None,
        orientation="z",
        radius=1.0,
        current=1.0,
        n_turns=1,
        mu=mu_0,
        srcType="inductive",
        **kwargs,
    ):
        if location is None:
            location = np.r_[0.0, 0.0, 0.0]

        if "moment" in kwargs:
            kwargs.pop("moment")

        # Raise error on deprecated arguments
        if (key := "N") in kwargs.keys():
            raise TypeError(f"'{key}' property has been removed. Please use 'n_turns'.")
        self.n_turns = n_turns

        BaseTDEMSrc.__init__(
            self, receiver_list=receiver_list, location=location, moment=None, **kwargs
        )

        self.orientation = orientation
        self.radius = radius
        self.current = current
        self.mu = mu
        self.srcType = srcType

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
        rad = validate_float("radius", rad, min_val=0, inclusive_min=False)
        self._radius = rad

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
        I = validate_float("current", I)
        if np.abs(I) == 0.0:
            raise ValueError("current must be non-zero.")
        self._current = I

    @property
    def moment(self):
        r"""Dipole moment of the loop.

        The dipole moment is given by :math:`NI\pi r^2`

        Returns
        -------
        float
            Dipole moment of the loop
        """
        return np.pi * self.radius**2 * self.current * self.n_turns

    @moment.setter
    def moment(self, value):
        if value is not None:
            warnings.warn(
                "Moment is not set as a property. I is the product"
                "of the loop radius and transmitter current",
                stacklevel=2,
            )
        pass

    @property
    def n_turns(self):
        """Number of turns in the loop.

        Returns
        -------
        int
        """
        return self._n_turns

    @n_turns.setter
    def n_turns(self, value):
        self._n_turns = validate_integer("n_turns", value, min_val=1)

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
        return self.n_turns * self._loop.vector_potential(obsLoc, coordinates)

    N = deprecate_property(
        n_turns, "N", "n_turns", removal_version="0.19.0", error=True
    )


class LineCurrent(BaseTDEMSrc):
    """Line current source.

    Given the wire path provided by the (n_loc, 3) locations array,
    the cells intersected by the wire path are identified and integrated
    source terms are computed.

    Parameters
    ----------
    receiver_list : list of simpeg.electromagnetics.time_domain.receivers.BaseRx
        List of TDEM receivers
    location : (n, 3) numpy.ndarray
        Array defining the node locations for the wire path. For inductive sources,
        you must close the loop, (i.e. provide the same point as the first and last
        entry of the array).
    current : float, optional
        A non-zero current value.
    mu : float, optional
        Magnetic permeability to use.
    """

    def __init__(
        self,
        receiver_list=None,
        location=None,
        current=1.0,
        mu=mu_0,
        srcType=None,
        **kwargs,
    ):
        super().__init__(receiver_list=receiver_list, location=location, **kwargs)
        for rx in self.receiver_list:
            if getattr(rx, "use_source_receiver_offset", False):
                raise ValueError(
                    "use_source_receiver_offset is ambiguous for a line current and is "
                    "not supported."
                )

        self.integrate = False
        self.current = current
        self.mu = mu

    @property
    def location(self):
        """Line current nodes locations

        Returns
        -------
        (n, 3) numpy.ndarray
            Line current node locations.
        """
        return self._location

    @location.setter
    def location(self, loc):
        loc = validate_ndarray_with_shape("location", loc, shape=("*", 3), dtype=float)
        if np.all(np.isclose(loc[0, :], loc[-1, :])):
            self._srcType = "inductive"
        else:
            self._srcType = "galvanic"
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
        I = validate_float("current", I)
        if np.abs(I) == 0.0:
            raise ValueError("current must be non-zero.")
        self._current = I

    @property
    def n_segments(self):
        """
        The number of line current segments.

        Returns
        -------
        int
        """
        return self.location.shape[0] - 1

    def Mejs(self, simulation):
        """Integrated electrical source term on edges

        Parameters
        ----------
        simulation : simpeg.electromagnetics.time_domain.simulation.BaseTDEMSimulation
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
        simulation : simpeg.electromagnetics.time_domain.simulation.BaseTDEMSimulation
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
        simulation : simpeg.electromagnetics.time_domain.simulation.BaseTDEMSimulation
            Base TDEM simulation

        Returns
        -------
        numpy.ndarray
            Right-hand side of galvanic source term. On edges for 'EB' formulation,
            and on faces for 'HJ' formulation.
        """
        if simulation._formulation == "EB":
            Grad = simulation.mesh.nodal_gradient
            return Grad.T * self.Mejs(simulation)
        elif simulation._formulation == "HJ":
            Div = sdiag(simulation.mesh.cell_volumes) * simulation.mesh.face_divergence
            return Div * self.Mfjs(simulation)

    def phiInitial(self, simulation):
        """Initial scalar potential

        Returns the scalar potential at the initial time is static
        fields are present.

        Parameters
        ----------
        simulation : simpeg.electromagnetics.base.BaseEMSimulation
            An electromagnetic simulation

        Returns
        -------
        Zero or numpy.ndarray
            Returns :class:`Zero` if there are no initial fields.
            Returns a numpy.ndarray if there are initial fields.

        """
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
                return -simulation.mesh.nodal_gradient * phi
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
            Grad = simulation.mesh.nodal_gradient
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
                Div = (
                    sdiag(simulation.mesh.cell_volumes)
                    * simulation.mesh.face_divergence
                )
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
        if self.waveform.has_initial_fields is False:
            return Zero()
        elif simulation._formulation != "HJ":
            raise NotImplementedError

        phi = self.phiInitial(simulation)
        Div = sdiag(simulation.mesh.cell_volumes) * simulation.mesh.face_divergence

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

        vol = simulation.mesh.cell_volumes
        Div = sdiag(vol) * simulation.mesh.face_divergence
        return (
            simulation.mesh.edge_curl
            * simulation.MeMuI
            * simulation.mesh.edge_curl.T.tocsr()
            - Div.T.tocsr()
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

        b = self.bInitial(simulation)
        return simulation.MeMuI * b

    def hInitialDeriv(self, simulation, v, adjoint=False, f=None):
        """Compute derivative of intitial magnetic field times a vector

        Parameters
        ----------
        simulation : BaseTDEMSimulation
            A SimPEG TDEM simulation
        v : numpy.ndarray
            A vector
        adjoint : bool
            If ``True``, return the adjoint

        Returns
        -------
        numpy.ndarray
            Derivative of initial magnetic field times a vector
        """
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

        if self.waveform.has_initial_fields is False:
            return Zero()
        elif simulation._formulation != "HJ":
            raise NotImplementedError

        a = self._aInitial(simulation)
        return simulation.mesh.edge_curl.T * a

    def bInitialDeriv(self, simulation, v, adjoint=False, f=None):
        """Compute derivative of intitial magnetic flux density times a vector

        Parameters
        ----------
        simulation : BaseTDEMSimulation
            A SimPEG tDEM simulation
        v : numpy.ndarray
            A vector
        adjoint : bool
            If ``True``, return the adjoint

        Returns
        -------
        numpy.ndarray
            Derivative of initial magnetic flux density times a vector
        """
        if self.waveform.has_initial_fields is False:
            return Zero()
        elif simulation._formulation != "HJ":
            raise NotImplementedError

        if adjoint is True:
            return self._aInitialDeriv(
                simulation, simulation.mesh.edge_curl * v, adjoint=True
            )
        return simulation.mesh.edge_curl.T * self._aInitialDeriv(simulation, v)

    def s_m(self, simulation, time):
        """Returns :class:`Zero` for ``LineCurrent``"""
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
    def __init__(self, receiver_list=None, s_e=None, **kwargs):
        self.integrate = False
        kwargs.pop("srcType", None)
        super(RawVec_Grounded, self).__init__(
            receiver_list, srcType="galvanic", **kwargs
        )

        if s_e is not None:
            self._Mfjs = self._s_e = s_e

    # def getRHSdc(self, simulation):
    #     return sdiag(simulation.mesh.cell_volumes) * simulation.mesh.face_divergence * self._s_e

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
    #     Div = sdiag(simulation.mesh.cell_volumes) * simulation.mesh.face_divergence
    #     return -simulation.MfRhoI * (Div.T * phi)

    # def jInitialDeriv(self, simulation, v, adjoint=False):
    #     if simulation._fieldType not in ["j", "h"]:
    #         raise NotImplementedError

    #     if self.waveform.has_initial_fields is False:
    #         return Zero()

    #     phi = self.phiInitial(simulation)
    #     Div = sdiag(simulation.mesh.cell_volumes) * simulation.mesh.face_divergence

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

    #     vol = simulation.mesh.cell_volumes

    #     return (
    #         simulation.mesh.edge_curl * simulation.MeMuI * simulation.mesh.edge_curl.T
    #         - simulation.mesh.face_divergence.T
    #         * sdiag(1.0 / vol * simulation.mui)
    #         * simulation.mesh.face_divergence  # stabalizing term. See (Chen, Haber & Oldenburg 2002)
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
    #     return simulation.mesh.edge_curl.T * a

    # def bInitialDeriv(self, simulation, v, adjoint=False, f=None):
    #     if simulation._fieldType not in ["j", "h"]:
    #         raise NotImplementedError

    #     if self.waveform.has_initial_fields is False:
    #         return Zero()

    #     if adjoint is True:
    #         return self._aInitialDeriv(simulation, simulation.mesh.edge_curl * v, adjoint=True)
    #     return simulation.mesh.edge_curl.T * self._aInitialDeriv(simulation, v)

    # def s_e(self, simulation, time):
    #     # if simulation._fieldType == 'h':
    #     #     return simulation.Mf * self._s_e * self.waveform.eval(time)
    #     return self._s_e * self.waveform.eval(time)
