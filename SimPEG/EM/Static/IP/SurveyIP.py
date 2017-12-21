from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from SimPEG.EM.Base import BaseEMSurvey
from SimPEG.EM.Static.DC.SrcDC import BaseSrc
from SimPEG.EM.Static.DC.RxDC import BaseRx
from SimPEG.EM.Static.DC import Survey as SurveyDC


class Survey(SurveyDC):

    def __init__(self, srcList, **kwargs):
        self.srcList = srcList
        SurveyDC.__init__(self, srcList, **kwargs)

    def dpred(self, m=None, f=None):
        """
            Predicted data.

            .. math::
                d_\\text{pred} = Pf(m)
        """
        return self.prob.Jvec(m, m, f=f)


def from_dc_to_ip_survey(dc_survey):
    srcList = dc_survey.srcList
    srcList_ip = []
    for src in srcList:
        srcList_ip.append(src_ip)
    return ip_survey
