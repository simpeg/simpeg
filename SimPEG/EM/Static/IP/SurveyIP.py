from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from SimPEG.EM.Base import BaseEMSurvey
from SimPEG.EM.Static.DC.SrcDC import BaseSrc
from SimPEG.EM.Static.DC.RxDC import BaseRx
import SimPEG.EM.Static.DC.RxDC as Rx
import SimPEG.EM.Static.DC.SrcDC as Src
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


def from_dc_to_ip_survey(dc_survey, flag=None):
    srcList = dc_survey.srcList
    if flag == "2.5D":
        ip_survey = Survey([srcList])
    else:
        srcList_ip = []
        for src in srcList:
            for rx in src.rxList:
                if isinstance(rx, Rx.Pole):
                    rx_ip = Rx.Pole(rxList_ip, locs=rx.locs)
                elif isinstance(rx, Rx.Dipole):
                    rx_ip = Rx.Dipole(rxList_ip, locs=rx.locs)
                else:
                    raise NotImplementedError()
                rxList_ip.append(rx_ip)
            if isinstance(src, Src.Pole):
                src_ip = Src.Pole(rxList_ip, loc=src_ip.loc)
            elif isinstance(src, Src.Dipole):
                src_ip = Src.Dipole(rxList_ip)
            else:
                raise NotImplementedError()
            srcList_ip.append(src_ip)
        ip_survey = Survey(srcList_ip)
    return ip_survey
