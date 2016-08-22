from SimPEG import Maps, Survey, Utils, np, sp
from scipy.constants import mu_0
import re


class LinearSurvey(Survey.BaseSurvey):
    """Base Magnetics Survey"""

    rxLoc = None  #: receiver locations
    rxType = None  #: receiver type

    def __init__(self, srcField, **kwargs):
        self.srcField = srcField
        Survey.BaseSurvey.__init__(self, **kwargs)

    def eval(self, u):
        return u

    @property
    def nD(self):
        return self.prob.G.shape[0]

    @property
    def nRx(self):
        return self.srcField.rxList[0].locs.shape[0]


class SrcField(Survey.BaseSrc):
    """ Define the inducing field """

    param = None  #: Inducing field param (Amp, Incl, Decl)

    def __init__(self, rxList, **kwargs):
        super(SrcField, self).__init__(rxList, **kwargs)


class RxObs(Survey.BaseRx):
    """A station location must have be located in 3-D"""
    def __init__(self, locsXYZ, **kwargs):
        locs = locsXYZ
        assert locsXYZ.shape[1] == 3, 'locs must in 3-D (x,y,z).'
        super(RxObs, self).__init__(locs, 'tmi',
                                    storeProjections=False, **kwargs)

    @property
    def nD(self):
        """Number of data in the receiver."""
        return self.locs[0].shape[0]
