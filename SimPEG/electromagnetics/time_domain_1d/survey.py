from ..base_1d import BaseEM1DSurvey
import numpy as np
import properties
from empymod import filters


class EM1DSurveyTD(BaseEM1DSurvey):
    """
    Survey class for time-domain surveys. Used for 1D simulation
    as well as stitched 1D simulation.
    """


    def __init__(self, source_list=None, **kwargs):
        BaseEM1DSurvey.__init__(self, source_list, **kwargs)

        # Use Sin filter for frequency to time transform
        self.fftfilt = filters.key_81_CosSin_2009()


    @property
    def nD(self):
        """
        Returns the number of data.
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


    @property
    def lowpass_filter(self):
        """
            Low pass filter values
        """
        if getattr(self, '_lowpass_filter', None) is None:
            # self._lowpass_filter = butterworth_type_filter(
            #     self.frequency, self.high_cut_frequency
            # )

            self._lowpass_filter = (1+1j*(self.frequency/self.high_cut_frequency))**-1
            self._lowpass_filter *= (1+1j*(self.frequency/3e5))**-0.99
            # For actual butterworth filter

            # filter_frequency, values = butter_lowpass_filter(
            #     self.high_cut_frequency
            # )
            # lowpass_func = interp1d(
            #     filter_frequency, values, fill_value='extrapolate'
            # )
            # self._lowpass_filter = lowpass_func(self.frequency)

        return self._lowpass_filter
