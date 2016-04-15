import SimPEG
from SimPEG.EM.Base import BaseEMSurvey
from SimPEG import sp
from SimPEG.Utils import Zero, Identity
from RxDC import BaseRx
from SrcDC import BaseSrc

class Survey(BaseEMSurvey):
    rxPair  = BaseRx
    srcPair = BaseSrc

    def __init__(self, srcList, **kwargs):
        self.srcList = srcList
        BaseEMSurvey.__init__(self, srcList, **kwargs)




