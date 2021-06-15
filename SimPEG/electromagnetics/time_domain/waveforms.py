import numpy as np
import properties
from ...utils import setKwargs

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

    use_lowpass_filter = properties.Bool("Switch for low pass filter", default=False)

    high_cut_frequency = properties.Float(
        "High cut frequency for low pass filter (Hz)", default=210 * 1e3
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

    def evalDeriv(self, time):
        t = np.asarray(time, dtype=float)
        out = np.zeros_like(t)

        out[(t < self.offTime) & (t >= self.eps)] = -1.0 / self.offTime

        if out.ndim == 0:
            out = out.item()
        return out

    @property
    def time_nodes(self):
        return np.r_[0.0, self.offTime]


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

    def evalDeriv(self, time):
        t = np.asarray(time, dtype=float)
        out = np.zeros_like(t)

        p_1 = (t <= self.peakTime) & (t >= 0.0)
        out[p_1] = (
            self.a
            / self.peakTime
            * np.exp(-self.a * t[p_1] / self.peakTime)
            / (1.0 - np.exp(-self.a))
        )

        p_2 = (t > self.peakTime) & (t < self.offTime)
        out[p_2] = -1.0 / (self.offTime - self.peakTime)

        if out.ndim == 0:
            out = out.item()
        return out

    @property
    def time_nodes(self):
        return np.r_[0, self.peakTime, self.offTime]


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

    def evalDeriv(self, time):
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
        return np.r_[self.ramp_on, self.ramp_off]


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

    @property
    def time_nodes(self):
        return np.r_[0, self.peakTime, self.offTime]


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

    def evalDeriv(self, time):
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

        p_2 = (t >= self.ramp_off[0]) & (t < self.ramp_off[1])
        out[p_2] = -1.0 / (self.ramp_off[1] - self.ramp_off[0])

        if out.ndim == 0:
            out = out.item()
        return out

    @property
    def time_nodes(self):
        return np.r_[self.ramp_on, self.ramp_off]


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

    def evalDeriv(self, time):
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

        p_2 = (t >= self.ramp_off[0]) & (t < self.ramp_off[1])
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

    @property
    def time_nodes(self):
        return np.r_[self.ramp_on, self.ramp_off]


class PiecewiseLinearWaveform(BaseWaveform):

    times = properties.Array("Time for input currents", dtype=float)

    currents = properties.Array("Input currents", dtype=float)

    def __init__(self, times, currents, **kwargs):
        super().__init__(**kwargs)
        times = np.asarray(times)
        currents = np.asarray(currents)
        if len(times) != len(currents):
            raise ValueError("time array and current array must be the same length")
        # ensure it is a sorted list...
        ind_sort = np.argsort(times)
        self.times = times[ind_sort]
        self.currents = currents[ind_sort]

    def eval(self, time):
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

    def evalDeriv(self, time):
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
