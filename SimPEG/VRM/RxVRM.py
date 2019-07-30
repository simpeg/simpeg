from SimPEG import Survey
import properties

#########################################
# BASE RECEIVER CLASS FOR VRM
#########################################


class BaseRxVRM(Survey.BaseRx, properties.HasProperties):
    """Base VRM receiver class"""

    def __init__(self, locs, **kwargs):

        super(BaseRxVRM, self).__init__(
            locs, 'None', storeProjections=False, **kwargs
        )


#########################################
# POINT RECEIVER CLASS FOR VRM
#########################################

class Point(BaseRxVRM):
    """Point receiver"""

    times = properties.Array('Observation times', dtype=float)
    fieldType = properties.StringChoice(
        'Field type', choices=["h", "b", "dhdt", "dbdt"]
    )
    fieldComp = properties.StringChoice(
        'Component of response', choices=["x", "y", "z"]
    )

    # def __init__(self, locs, times, fieldType, fieldComp, **kwargs):
    def __init__(self, locs, **kwargs):

        if locs.shape[1] != 3:
            raise ValueError(
                'Rx locations (xi,yi,zi) must be np.array(N,3) where N is the number of stations'
            )

        super(Point, self).__init__(locs, **kwargs)

    @property
    def nTimes(self):
        """Number of measurements times."""
        if self.times is not None:
            return len(self.times)

    @property
    def nLocs(self):
        """Number of locations."""
        return self.locs.shape[0]

    @property
    def nD(self):
        """Number of data in the receiver."""
        if self.times is not None:
            return self.locs.shape[0] * len(self.times)


class SquareLoop(BaseRxVRM):
    """Square loop receiver

    Measurements with this type of receiver are the field, integrated over the
    area of the loop, then multiplied by the number of coils, then normalized
    by the dipole moment. As a result, the units for fields predicted with this
    type of receiver are the same as 'h', 'b', 'dhdt' and 'dbdt', respectively.

    """

    times = properties.Array('Observation times', dtype=float)
    width = properties.Float('Square loop width', min=1e-6)
    nTurns = properties.Integer('Number of loop turns', min=1, default=1)
    quadOrder = properties.Integer(
        'Order for numerical quadrature integration over loop', min=1, max=7, default=3
    )
    fieldType = properties.StringChoice('Field type', choices=["h", "b", "dhdt", "dbdt"])
    fieldComp = properties.StringChoice('Component of response', choices=["x", "y", "z"])

    def __init__(self, locs, **kwargs):

        if locs.shape[1] != 3:
            raise ValueError(
                'Rx locations (xi,yi,zi) must be np.array(N,3) where N is the number of stations'
            )

        super(SquareLoop, self).__init__(locs, **kwargs)

    @property
    def nTimes(self):
        """Number of measurements times."""
        if self.times is not None:
            return len(self.times)

    @property
    def nLocs(self):
        """Number of locations."""
        return self.locs.shape[0]

    @property
    def nD(self):
        """Number of data in the receiver."""
        if self.times is not None:
            return self.locs.shape[0] * len(self.times)
