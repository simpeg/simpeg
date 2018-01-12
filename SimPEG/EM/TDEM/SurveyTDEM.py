from __future__ import division, print_function
import SimPEG
from SimPEG import Utils
from SimPEG.Utils import Zero, Identity
from scipy.constants import mu_0
from SimPEG.EM.Utils import *
from . import SrcTDEM as Src
from . import RxTDEM as Rx


####################################################
# Survey
####################################################

class Survey(SimPEG.Survey.BaseSurvey):
    """
    Time domain electromagnetic survey
    """

    srcPair = Src.BaseTDEMSrc
    rxPair = Rx

    def __init__(self, srcList, **kwargs):
        # Sort these by frequency
        self.srcList = srcList
        SimPEG.Survey.BaseSurvey.__init__(self, **kwargs)

    def eval(self, u):
        data = SimPEG.Survey.Data(self)
        for src in self.srcList:
            for rx in src.rxList:
                data[src, rx] = rx.eval(src, self.mesh, self.prob.timeMesh, u)
        return data

    def evalDeriv(self, u, v=None, adjoint=False):
        raise Exception('Use Receivers to project fields deriv.')
