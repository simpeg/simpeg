import numpy as np
import properties
from SimPEG.survey import BaseRx, BaseTimeRx


class TimeDomainPointReceiver(BaseTimeRx):
    """
    Receiver class for simulating the time-domain magnetic response at a point.

    :param numpy.ndarray locations: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param numpy.array times: time channels [s]
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: data component 'h', 'b', 'dhdt' or 'dbdt'
    :param numpy.array frequencies: frequencies used to compute harmonic response
    :param numpy.array time_interval: on-times [s]
    :param bool use_source_receiver_offset: actual receiver location (False). Source-receiver offset (True)
    """

    ftarg = None

    locations = properties.Array(
        "Receiver Location", dtype=float, shape=("*",), required=True
    )

    # times property is inherited from BaseTimeRx class

    orientation = properties.StringChoice(
        "Field orientation", default="z", choices=["x", "y", "z"]
    )

    component = properties.StringChoice(
        "component of the field (h, b, dhdt, dbdt)",
        default="dbdt",
        choices=["h", "b", "dhdt", "dbdt"]
    )

    frequencies = properties.Array(
        "Frequency (Hz)", dtype=float, shape=("*",), required=True
    )

    time_interval = properties.Array(
        "Full time interval", dtype=float, shape=("*",)
    )

    use_source_receiver_offset = properties.Bool(
        "Use source-receiver offset",
        default=False
    )

    times_dual_moment = properties.Array(
        "Off-time channels (s) for the dual moment", dtype=float
    )

    def __init__(self, locations=None, times=None, orientation=None, component=None, use_source_receiver_offset=None, **kwargs):

        super(TimeDomainPointReceiver, self).__init__(locations, times, **kwargs)

        if orientation is not None:
            self.orientation = orientation
        if component is not None:
            self.component = component
        if use_source_receiver_offset is not None:
            self.use_source_receiver_offset = use_source_receiver_offset

        # Required static property
        self.field_type = "secondary"


    @property
    def n_time(self):
        """
            Number of time channels
        """
        if self.times_dual_moment is not None:
            return int(self.times.size) + int(self.times_dual_moment.size)
        else:
            return int(self.times.size)

    @property
    def n_frequency(self):
        """
            Number of frequencies
        """
        return int(self.frequencies.size)

    @property
    def nD(self):
        """Number of data in the receiver."""
        return self.n_time

