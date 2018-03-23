from SimPEG import Survey


#########################################
# BASE VRM RECEIVER CLASS
#########################################

class BaseRxVRM(Survey.BaseRx):
    """BaseRxVRM class"""

    def __init__(self, locs, times, **kwargs):
        assert locs.shape[1] == 3, 'locs must in 3-D (x,y,z).'
        super(BaseRxVRM, self).__init__(locs, 'None', storeProjections=False, **kwargs)
        self.times = times

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


#########################################
# H AT POINT CLASS
#########################################

class Point_h(BaseRxVRM):
    """

    """

    def __init__(self, locsXYZ, times, fieldComp, **kwargs):
        BaseRxVRM.__init__(self, locsXYZ, times, **kwargs)
        self.fieldType = 'h'
        self.fieldComp = fieldComp


#########################################
# dH/dt AT POINT CLASS
#########################################

class Point_dhdt(BaseRxVRM):
    """

    """

    def __init__(self, locsXYZ, times, fieldComp, **kwargs):
        BaseRxVRM.__init__(self, locsXYZ, times, **kwargs)
        self.fieldType = 'dhdt'
        self.fieldComp = fieldComp


#########################################
# B AT POINT CLASS
#########################################

class Point_b(BaseRxVRM):
    """

    """

    def __init__(self, locsXYZ, times, fieldComp, **kwargs):
        BaseRxVRM.__init__(self, locsXYZ, times, **kwargs)
        self.fieldType = 'b'
        self.fieldComp = fieldComp


#########################################
# dB/dt AT POINT CLASS
#########################################

class Point_dbdt(BaseRxVRM):

    """

    """

    def __init__(self, locsXYZ, times, fieldComp, **kwargs):
        BaseRxVRM.__init__(self, locsXYZ, times, **kwargs)
        self.fieldType = 'dbdt'
        self.fieldComp = fieldComp
