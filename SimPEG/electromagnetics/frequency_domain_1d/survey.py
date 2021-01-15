from ..base_1d import BaseEM1DSurvey
import numpy as np
import properties


class EM1DSurveyFD(BaseEM1DSurvey):
    """
    Survey class for frequency domain surveys. Used for 1D simulation
    as well as stitched 1D simulation.
    """

    def __init__(self, source_list=None, **kwargs):
        BaseEM1DSurvey.__init__(self, source_list, **kwargs)

    @property
    def survey_type(self):
        return 'frequency_domain'
    

    @property
    def nD(self):
        """
        Returns number of data.
        """

        nD = 0

        for src in self.source_list:
            for rx in src.receiver_list:
                nD += rx.nD

        return int(nD)

    @property
    def vnD_by_sounding(self):
        if getattr(self, '_vnD_by_sounding', None) is None:
            temp = []
            for src in self.source_list:
                temp.append(
                    np.sum([rx.nD for rx in src.receiver_list])
                )
            self._vnD_by_sounding = np.array(temp)
        return self._vnD_by_sounding

