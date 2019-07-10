from ...survey import BaseRx
import properties

import warnings


#########################################
# POINT RECEIVER CLASS FOR VRM
#########################################

class Point(BaseRx):
    """Point receiver"""

    times = properties.Array('Observation times', dtype=float)

    fieldType = properties.StringChoice(
        'Field type', choices=["h", "b", "dhdt", "dbdt"]
    )

    orientation = properties.StringChoice(
        'Component of response', choices=["x", "y", "z"]
    )

    def __init__(self, locations=None, **kwargs):

        if locations.shape[1] != 3:
            raise ValueError(
                'Rx locations (xi,yi,zi) must be np.array(N,3) where N is the number of stations'
            )

        super(Point, self).__init__(locations, **kwargs)

    @property
    def n_times(self):
        """Number of measurements times."""
        if self.times is not None:
            return len(self.times)

    @property
    def nTimes(self):
        warnings.warn(
            "Point.nTimes will be depreciated in favour of "
            "Point.n_times. Please update your code accordingly"
        )
        return self.n_times

    @property
    def n_locations(self):
        """Number of locations."""
        return self.locs.shape[0]

    @property
    def nLocs(self):
        warnings.warn(
            "Point.nLocs will be depreciated in favour of "
            "Point.n_times. Please update your code accordingly"
        )
        return self.n_times

    @property
    def nD(self):
        """Number of data in the receiver."""
        if self.times is not None:
            return self.locations.shape[0] * len(self.times)

    @property
    def fieldComp(self):
        warnings.warn(
            "Point.fieldComp will be depreciated in favour of "
            "Point.orientation. Please update your code accordingly"
        )
        return self.orientation

    @fieldComp.setter
    def fieldComp(self, value):
        warnings.warn(
            "Point.fieldComp will be depreciated in favour of "
            "Point.orientation. Please update your code accordingly"
        )
        self.orientation = value



class SquareLoop(Point):
    """Square loop receiver

    Measurements with this type of receiver are the field, integrated over the
    area of the loop, then multiplied by the number of coils, then normalized
    by the dipole moment. As a result, the units for fields predicted with this
    type of receiver are the same as 'h', 'b', 'dhdt' and 'dbdt', respectively.

    """

    width = properties.Float('Square loop width', min=1e-6)
    nTurns = properties.Integer('Number of loop turns', min=1, default=1)
    quadOrder = properties.Integer(
        'Order for numerical quadrature integration over loop', min=1, max=7, default=3
    )

    def __init__(self, locations, **kwargs):

        if locations.shape[1] != 3:
            raise ValueError(
                'Rx locations (xi,yi,zi) must be np.array(N,3) where N is the number of stations'
            )

        super(SquareLoop, self).__init__(locations, **kwargs)
