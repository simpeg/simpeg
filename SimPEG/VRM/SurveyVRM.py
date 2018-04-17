import SimPEG
from SimPEG import Survey
from SimPEG.VRM import RxVRM, SrcVRM
import numpy as np
import properties

############################################
# BASE VRM SURVEY CLASS
############################################

class SurveyVRM(Survey.BaseSurvey, properties.HasProperties):

    """

    """

    t_active = properties.Array('Boolean array where True denotes active data in the inversion', dtype=bool)

    def __init__(self, srcList, **kwargs):
        super(SurveyVRM, self).__init__(**kwargs)
        self.srcList = srcList
        self.srcPair = SrcVRM.BaseSrcVRM
        self.rxPair = RxVRM.BaseRxVRM
        if self.t_active is None:
            self.t_active = np.ones(self.nD, dtype=bool)

    @properties.observer('t_active')
    def _t_active_observer(self, change):
        print("Indicies of active data changed and may not correspond to property: t_interval")

    def set_active_interval(self, tmin, tmax):
        """Set active times using an interval"""

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

        self.t_active = (tActBool >= tmin) & (tActBool <= tmax)

    def dpred(self, m=None, f=None):

        """

        """

        assert self.ispaired, "Survey must be paired with a VRM problem"

        if f is not None:

            return f[self.t_active]

        elif f is None and isinstance(self.prob, SimPEG.VRM.ProblemVRM.Problem_Linear):

            f = self.prob.fields(m)

            return f[self.t_active]

        elif f is None and isinstance(self.prob, SimPEG.VRM.ProblemVRM.Problem_LogUniform):

            f = self.prob.fields()

            return f[self.t_active]
