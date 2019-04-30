from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from SimPEG.EM.Base import BaseEMSurvey
from SimPEG.EM.Static.DC import Survey as SurveyDC
from SimPEG.EM.Static import DC


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


def from_dc_to_ip_survey(dc_survey, dim="2.5D"):
    srcList = dc_survey.srcList

    # for 2.5D
    if dim == "2.5D":
        srcList_ip = []
        for src in srcList:
            rxList_ip = []
            src_ip = []
            for rx in src.rxList:
                if isinstance(rx, DC.Rx.Pole_ky):
                    rx_ip = DC.Rx.Pole(rx.locs)
                elif isinstance(rx, DC.Rx.Dipole_ky):
                    rx_ip = DC.Rx.Dipole(rx.locs[0], rx.locs[1])
                else:
                    print(rx)
                    raise NotImplementedError()
                rxList_ip.append(rx_ip)

            if isinstance(src, DC.Src.Pole):
                src_ip = DC.Src.Pole(
                    rxList_ip, src_ip.loc
                )
            elif isinstance(src, DC.Src.Dipole):
                src_ip = DC.Src.Dipole(
                    rxList_ip, src.loc[0], src.loc[1]
                )
            else:
                print(src)
                raise NotImplementedError()
            srcList_ip.append(src_ip)

        ip_survey = Survey(srcList_ip)

    # for 2D or 3D case
    elif (dim == "2D") or ("3D"):
        ip_survey = Survey(srcList)

    else:
        raise Exception(" dim must be '2.5D', '2D', or '3D' ")

    return ip_survey
