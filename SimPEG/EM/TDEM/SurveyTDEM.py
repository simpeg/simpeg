from ...survey import BaseSurvey
from ...data import Data
from . import SrcTDEM as Src
from . import RxTDEM as Rx


####################################################
# Survey
####################################################

class Survey(BaseSurvey):
    """
    Time domain electromagnetic survey
    """

    srcPair = Src.BaseTDEMSrc
    rxPair = Rx

    def __init__(self, srcList, **kwargs):
        # Sort these by frequency
        self.srcList = srcList
        BaseSurvey.__init__(self, **kwargs)

    def eval(self, u):
        data = Data(self)
        for src in self.srcList:
            for rx in src.rxList:
                data.dobs[src, rx] = rx.eval(src, self.mesh, self.prob.timeMesh, u)
        return data

    def evalDeriv(self, u, v=None, adjoint=False):
        raise Exception('Use Receivers to project fields deriv.')
