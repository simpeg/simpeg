from ...survey import BaseRx
import properties
from ...utils.code_utils import deprecate_property

import warnings


#########################################
# POINT RECEIVER CLASS FOR VRM
#########################################


class Point(BaseRx):
    """Point receiver"""

    times = properties.Array("Observation times", dtype=float)

    fieldType = properties.StringChoice(
        "Field type", choices=["h", "b", "dhdt", "dbdt"]
    )

    orientation = properties.StringChoice(
        "Component of response", choices=["x", "y", "z"]
    )

    def __init__(self, locations=None, **kwargs):

        if locations.shape[1] != 3:
            raise ValueError(
                "Rx locations (xi,yi,zi) must be np.array(N,3) where N is the number of stations"
            )

        super(Point, self).__init__(locations, **kwargs)

    @property
    def n_times(self):
        """Number of measurements times."""
        if self.times is not None:
            return len(self.times)

    nTimes = deprecate_property(
        n_times, "nTimes", new_name="n_times", removal_version="0.15.0"
    )

    @property
    def n_locations(self):
        """Number of locations."""
        return self.locations.shape[0]

    nLocs = deprecate_property(
        n_locations, "nLocs", new_name="n_locations", removal_version="0.15.0"
    )

    @property
    def nD(self):
        """Number of data in the receiver."""
        if self.times is not None:
            return self.locations.shape[0] * len(self.times)

    fieldComp = deprecate_property(
        orientation, "fieldComp", new_name="orientation", removal_version="0.15.0"
    )


class SquareLoop(Point):
    """Square loop receiver

    Measurements with this type of receiver are the field, integrated over the
    area of the loop, then multiplied by the number of coils, then normalized
    by the dipole moment. As a result, the units for fields predicted with this
    type of receiver are the same as 'h', 'b', 'dhdt' and 'dbdt', respectively.

    """

    width = properties.Float("Square loop width", min=1e-6)
    nTurns = properties.Integer("Number of loop turns", min=1, default=1)
    quadOrder = properties.Integer(
        "Order for numerical quadrature integration over loop", min=1, max=7, default=3
    )

    def __init__(self, locations, **kwargs):

        if locations.shape[1] != 3:
            raise ValueError(
                "Rx locations (xi,yi,zi) must be np.array(N,3) where N is the number of stations"
            )

        super(SquareLoop, self).__init__(locations, **kwargs)
