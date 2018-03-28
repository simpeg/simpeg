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
