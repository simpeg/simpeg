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

    def __init__(self, receiver_list=None, location=None, waveform=None, **kwargs):
        super(BaseTimeSrc, self).__init__(
            receiver_list=receiver_list, location=location, **kwargs
        )
        self.waveform = waveform


    

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


class MagneticDipoleSource(BaseTimeSrc):
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

    def __init__(self, receiver_list=None, location=None, waveform=None, **kwargs):
        super(MagneticDipoleSource, self).__init__(
            receiver_list=receiver_list, location=location, waveform=waveform, **kwargs
        )


class HorizontalLoopSource(BaseTimeSrc):
    """
    Time-domain horizontal loop source.

    :param numpy.array location: source location (x,y,z)
    :param float I: source current amplitude [A]
    :param float a: loop radius [m]
    """

    I = properties.Float("Source loop current", default=1.)

    a = properties.Float("Source loop radius", default=np.sqrt(1/np.pi))

    def __init__(self, receiver_list=None, location=None, waveform=None,  **kwargs):
        super(HorizontalLoopSource, self).__init__(
            receiver_list=receiver_list, location=location, waveform=waveform, **kwargs
        )


class LineCurrentSource(BaseTimeSrc):
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

    def __init__(self, receiver_list=None, location=None, waveform=None,  **kwargs):
        super(LineCurrentSource, self).__init__(
            receiver_list=receiver_list, location=location, waveform=waveform, **kwargs
        )


