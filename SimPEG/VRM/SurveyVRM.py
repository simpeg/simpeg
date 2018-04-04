import SimPEG
from SimPEG import Survey
from SimPEG.VRM import RxVRM, SrcVRM
import numpy as np


############################################
# BASE VRM SURVEY CLASS
############################################

class SurveyVRM(Survey.BaseSurvey):

    """

    """

    _tActive = None
    _tActIsSet = False
    _tInterval = None

    def __init__(self, srcList, **kwargs):
        super(SurveyVRM, self).__init__(**kwargs)
        self.srcList = srcList
        self.srcPair = SrcVRM.BaseSrcVRM
        self.rxPair = RxVRM.BaseRxVRM

        self._tInterval = kwargs.get('ActiveTimeInterval', [-np.inf, np.inf])
        assert isinstance(self._tInterval, list), "Active time interval must be a list of length 2 (i.e. [t1, t2])"
        assert len(self._tInterval) == 2, "Active time interval must be a list of length 2 (i.e. [t1, t2])"

    @property
    def ActiveTimeInterval(self):
        return self._tInterval

    @ActiveTimeInterval.setter
    def ActiveTimeInterval(self, List):
        assert isinstance(List, list), "Active time interval must be a list of length 2 (i.e. [t1, t2])"
        assert len(List) == 2, "Active time interval must be a list of length 2 (i.e. [t1, t2])"
        self._tActIsSet = False
        self._tInterval = List

    @property
    def tActive(self):
        if self._tActIsSet:
            return self._tActive
        else:
            srcList = self.srcList
            nSrc = len(srcList)
            tActBool = np.array([])

            for pp in range(0, nSrc):

                rxList = srcList[pp].rxList
                nRx = len(rxList)

                for qq in range(0, nRx):

                    times = rxList[qq].times
                    nLoc = np.shape(rxList[qq].locs)[0]
                    tActBool = np.r_[tActBool, np.kron(np.ones(nLoc), times)]

            self._tActIsSet = True
            self.tActive = (tActBool >= self._tInterval[0]) & (tActBool <= self._tInterval[1])
            return self._tActive

    @tActive.setter
    def tActive(self, BoolArgs):

        assert len(BoolArgs) == self.nD, "Must be an array or list of boolean arguments with length equal to total number of data"
        print('SETTING NEW ACTIVE TIMES')
        self._tActIsSet = True
        self._tActive = BoolArgs

    def dpred(self, m=None, f=None):

        """

        """

        assert self.ispaired, "Survey must be paired with a VRM problem"

        if f is not None:

            return f[self.tActive]

        elif f is None and isinstance(self.prob, SimPEG.VRM.ProblemVRM.Problem_Linear):

            f = self.prob.fields(m)

            return f[self.tActive]

        elif f is None and isinstance(self.prob, SimPEG.VRM.ProblemVRM.Problem_LogUniform):

            f = self.prob.fields()

            return f[self.tActive]
