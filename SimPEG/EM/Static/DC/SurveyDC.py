from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from future import standard_library
standard_library.install_aliases()
import SimPEG
from SimPEG.EM.Base import BaseEMSurvey
from SimPEG import sp, Survey
from SimPEG.Utils import Zero, Identity
from .RxDC import BaseRx
from .SrcDC import BaseSrc

class Survey(BaseEMSurvey):
    rxPair  = BaseRx
    srcPair = BaseSrc

    def __init__(self, srcList, **kwargs):
        self.srcList = srcList
        BaseEMSurvey.__init__(self, srcList, **kwargs)

class Survey_ky(BaseEMSurvey):
    rxPair  = BaseRx
    srcPair = BaseSrc

    def __init__(self, srcList, **kwargs):
        self.srcList = srcList
        BaseEMSurvey.__init__(self, srcList, **kwargs)

    def eval(self, f):
        """
        Project fields to receiver locations
        :param Fields u: fields object
        :rtype: numpy.ndarray
        :return: data
        """
        data = SimPEG.Survey.Data(self)
        kys = self.prob.kys
        for src in self.srcList:
            for rx in src.rxList:
                data[src, rx] = rx.eval(kys, src, self.mesh, f)
        return data


