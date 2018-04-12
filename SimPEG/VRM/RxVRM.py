from SimPEG import Survey
import properties

#########################################
# BASE RECEIVER CLASS FOR VRM
#########################################

class BaseRxVRM(Survey.BaseRx):
    """Base VRM receiver class"""

    def __init__(self, locs, **kwargs):

        super(BaseRxVRM, self).__init__(locs, 'None', storeProjections=False, **kwargs)


#########################################
# POINT RECEIVER CLASS FOR VRM
#########################################

class Point(BaseRxVRM):
    """Point receiver"""

    def __init__(self, locs, times, fieldType, fieldComp, **kwargs):
        assert locs.shape[1] == 3, 'locs must in 3-D (x,y,z).'
        super(Point, self).__init__(locs, **kwargs)
        self.times = times
        assert fieldType in ['h', 'b', 'dhdt', 'dbdt'], '"fieldType" must be one of "h", "b", "dhdt" or "dbdt"'
        self.fieldType = fieldType
        assert fieldComp in ['x', 'y', 'z'], '"fieldComp" must be one of "x", "y" or "z"'
        self.fieldComp = fieldComp

    @property
    def nTimes(self):
        """Number of measurements times."""
        return len(self.times)

    @property
    def nLocs(self):
        """Number of locations."""
        return self.locs.shape[0]

    @property
    def nD(self):
        """Number of data in the receiver."""
        return self.locs.shape[0] * len(self.times)


class SquareLoop(BaseRxVRM):
    """Square loop receiver

    Measurements with this type of receiver are the field, integrated over the
    area of the loop, then multiplied by the number of coils, then normalized
    by the dipole moment. As a result, the units for fields predicted with this
    type of receiver are the same as 'h', 'b', 'dhdt' and 'dbdt', respectively.

    """

    _quad_order = None

    def __init__(self, locs, times, width, nTurns, fieldType, fieldComp, **kwargs):

        self._quad_order = kwargs.get('quad_order', 4)

        assert locs.shape[1] == 3, 'locs must in 3-D (x,y,z).'
        super(SquareLoop, self).__init__(locs, **kwargs)
        self.times = times
        assert isinstance(width, float), "Side length of square loop must be a float"
        self.width = width
        assert isinstance(nTurns, int), "Number of turns must be an integer"
        self.nTurns = nTurns
        assert fieldType in ['h', 'b', 'dhdt', 'dbdt'], '"fieldType" must be one of "h", "b", "dhdt" or "dbdt"'
        self.fieldType = fieldType
        assert fieldComp in ['x', 'y', 'z'], '"fieldComp" must be one of "x", "y" or "z"'
        self.fieldComp = fieldComp

    @property
    def nTimes(self):
        """Number of measurements times."""
        return len(self.times)

    @property
    def nLocs(self):
        """Number of locations."""
        return self.locs.shape[0]

    @property
    def nD(self):
        """Number of data in the receiver."""
        return self.locs.shape[0] * len(self.times)

    @property
    def quad_order(self):
        """Order for evaluation of integral using Gaussian quadrature"""
        return self._quad_order

    @quad_order.setter
    def quad_order(self, Val):

        assert isinstance(Val, int) and Val > 0, "Order of Gaussian quadrature must be an integer"
        assert Val < 8 and Val > 0, "Order of Gaussian quadrature must be a number from 1 to 7"

        self._quad_order = Val
