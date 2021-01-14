import numpy as np
from SimPEG import survey
import properties
from scipy import special as spec


#############################################################################
# Time Sources
#############################################################################


class BaseTimeSrc(survey.BaseSrc):
    """
    Base class for EM1D time-domain sources.

    :param numpy.array location: source location (x,y,z)
    :param string orientation: dipole orientation 'x', 'y' or 'z'
    :param float moment_amplitude: magnitude of the dipole moment |m|
    """

    wave_type = properties.StringChoice(
        "Waveform",
        default="stepoff",
        choices=["stepoff", "general"]
    )

    moment_type = properties.StringChoice(
        "Source moment type",
        default="single",
        choices=["single", "dual"]
    )

    n_pulse = properties.Integer(
        "The number of pulses",
    )

    base_frequency = properties.Float(
        "Base frequency (Hz)"
    )

    time_input_currents = properties.Array(
        "Time for input currents", dtype=float
    )

    input_currents = properties.Array(
        "Input currents", dtype=float
    )

    use_lowpass_filter = properties.Bool(
        "Switch for low pass filter", default=False
    )

    high_cut_frequency = properties.Float(
        "High cut frequency for low pass filter (Hz)",
        default=210*1e3
    )


    # ------------- For dual moment ------------- #

    time_input_currents_dual_moment = properties.Array(
        "Time for input currents (dual moment)", dtype=float
    )

    input_currents_dual_moment = properties.Array(
        "Input currents (dual moment)", dtype=float
    )

    base_frequency_dual_moment = properties.Float(
        "Base frequency for the dual moment (Hz)"
    )


    def __init__(self, receiver_list=None, **kwargs):
        super(BaseTimeSrc, self).__init__(receiver_list=receiver_list, **kwargs)


    @property
    def period(self):
        return 1./self.base_frequency

    @property
    def pulse_period(self):
        Tp = (
            self.time_input_currents.max() -
            self.time_input_currents.min()
        )
        return Tp

    # ------------- For dual moment ------------- #
    @property
    def period_dual_moment(self):
        return 1./self.base_frequency_dual_moment

    @property
    def pulse_period_dual_moment(self):
        Tp = (
            self.time_input_currents_dual_moment.max() -
            self.time_input_currents_dual_moment.min()
        )
        return Tp

    # Note: not relevant here
    # @property
    # def n_time_dual_moment(self):
    #     return int(self.time_dual_moment.size)

    # @property
    # def nD(self):
    #     """
    #         # of data
    #     """

    #     if self.moment_type == "single":
    #         return self.n_time
    #     else:
    #         return self.n_time + self.n_time_dual_moment


class TimeDomainMagneticDipoleSource(BaseTimeSrc):
    """
    Time-domain magnetic dipole source.

    :param numpy.array location: source location (x,y,z)
    :param string orientation: dipole orientation 'z'
    :param float moment_amplitude: magnitude of the dipole moment |m|
    """

    moment_amplitude = properties.Float("Magnitude of the dipole moment", default=1.)

    orientation = properties.StringChoice(
        "Dipole Orientation", default="z", choices=["z"]
    )

    def __init__(self, receiver_list=None, **kwargs):
        super(TimeDomainMagneticDipoleSource, self).__init__(receiver_list=receiver_list, **kwargs)


class TimeDomainHorizontalLoopSource(BaseTimeSrc):
    """
    Time-domain horizontal loop source.

    :param numpy.array location: source location (x,y,z)
    :param float I: source current amplitude [A]
    :param float a: loop radius [m]
    """

    I = properties.Float("Source loop current", default=1.)

    a = properties.Float("Source loop radius", default=1.)

    def __init__(self, receiver_list=None, **kwargs):
        super(TimeDomainHorizontalLoopSource, self).__init__(receiver_list=receiver_list, **kwargs)


class TimeDomainLineSource(BaseTimeSrc):
    """
    Time-domain current line source.

    :param numpy.ndarray node_locations: np.array(N+1, 3) of node locations defining N line segments
    :param float I: current amplitude [A]
    """

    I = properties.Float("Source loop current", default=1.)

    node_locations = properties.Array(
        "Source path (xi, yi, zi), i=0,...N",
        dtype=float
    )

    def __init__(self, receiver_list=None, **kwargs):
        super(TimeDomainLineSource, self).__init__(receiver_list=receiver_list, **kwargs)


