import SimPEG
from SimPEG.EM.Base import BaseEMSurvey
from SimPEG import np, sp, Survey
from SimPEG.Utils import Zero, Identity
from SimPEG.EM.Static.DC.SrcDC import BaseSrc
from SimPEG.EM.Static.SIP.RxSIP import BaseRx

class Survey(BaseEMSurvey):
    rxPair  = BaseRx
    srcPair = BaseSrc
    times = None

    def __init__(self, srcList, **kwargs):
        self.srcList = srcList
        BaseEMSurvey.__init__(self, srcList, **kwargs)
        self.getUniqueTimes()

    def getUniqueTimes(self):
        time_rx = []
        for src in self.srcList:
            for rx in src.rxList:
                time_rx.append(rx.times)
        self.times = np.unique(np.hstack(time_rx))

    def dpred(self, m, f=None):
        """
            Predicted data.

            .. math::
                d_\\text{pred} = Pf(m)
        """
        return self.prob.Jvec(m, m, f=f)
