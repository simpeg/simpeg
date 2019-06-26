from ..resistivity import Survey as SurveyDC
from ..resistivity import Rx, Src


class Survey(SurveyDC):

    def __init__(self, srcList, **kwargs):
        self.srcList = srcList
        Survey__init__(self, srcList, **kwargs)

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
                if isinstance(rx, Rx.Pole_ky):
                    rx_ip = Rx.Pole(rx.locs)
                elif isinstance(rx, Rx.Dipole_ky):
                    rx_ip = Rx.Dipole(rx.locs[0], rx.locs[1])
                else:
                    print(rx)
                    raise NotImplementedError()
                rxList_ip.append(rx_ip)

            if isinstance(src, Src.Pole):
                src_ip = Src.Pole(
                    rxList_ip, src_ip.loc
                )
            elif isinstance(src, Src.Dipole):
                src_ip = Src.Dipole(
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
