import numpy as np
import properties

from ....survey import BaseTimeSurvey
from . import sources
from . import receivers
from .. import resistivity as dc


class Survey(BaseTimeSurvey):
    """
    Spectral induced polarization survey
    """

    n_pulse = 2
    T = 8.0

    source_list = properties.List(
        "A list of sources for the survey",
        properties.Instance("A SimPEG source", sources.BaseSrc),
        default=[],
    )

    def __init__(self, source_list=None, **kwargs):
        super().__init__(source_list, **kwargs)

    @property
    def n_locations(self):
        return int(self.nD / self.times.size)


def from_dc_to_sip_survey(survey_dc, times):
    """
    Generate sip survey from dc survey
    """
    srcList = survey_dc.source_list

    srcList_sip = []
    for src in srcList:
        rxList_sip = []
        for rx in src.receiver_list:
            if isinstance(rx, dc.receivers.Pole):
                rx_sip = receivers.Pole(rx.locations, times=times)
            elif isinstance(rx, dc.receivers.Dipole):
                rx_sip = receivers.Dipole(rx.locations[0], rx.locations[1], times=times)
            else:
                print(rx)
                raise NotImplementedError()
            rxList_sip.append(rx_sip)

        if isinstance(src, dc.sources.Pole):
            src_sip = sources.Pole(rxList_sip, src.loc)
        elif isinstance(src, dc.sources.Dipole):
            src_sip = sources.Dipole(rxList_sip, src.loc[0], src.loc[1])
        else:
            print(src)
            raise NotImplementedError()
        srcList_sip.append(src_sip)

    survey_sip = Survey(srcList_sip)

    return survey_sip
