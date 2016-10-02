import SimPEG
from SimPEG.EM.Base import BaseEMSurvey
from SimPEG import sp, Survey
from SimPEG.Utils import Zero, Identity
from SimPEG.EM.Static.DC.SrcDC import BaseSrc
from SimPEG.EM.Static.DC.RxDC import BaseRx

class Survey(BaseEMSurvey):
    rxPair  = BaseRx
    srcPair = BaseSrc

    def __init__(self, srcList, **kwargs):
        self.srcList = srcList
        BaseEMSurvey.__init__(self, srcList, **kwargs)

    def dpred(self, m, f=None):
        """
            Predicted data.

            .. math::
                d_\\text{pred} = Pf(m)
        """
        return self.prob.Jvec(m, m, f=f)
