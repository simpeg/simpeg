import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import mu_0
from .supporting_functions.waveform_functions import *
import properties

############################################################
#               BASE WAVEFORM CLASSES
############################################################

class StepoffWaveform(properties.HasProperties):
    """Waveform class for a unit stepoff function"""

    use_lowpass_filter = properties.Bool(
        "Switch for low pass filter", default=False
    )

    n_pulse = properties.Integer(
        "The number of pulses",
        default=1
    )

    high_cut_frequency = properties.Float(
        "High cut frequency for low pass filter (Hz)",
        default=210*1e3
    )

    def __init__(self, **kwargs):
        super(StepoffWaveform, self).__init__()

    @property
    def wave_type(self):
        return "stepoff"


class GeneralWaveform(StepoffWaveform):
    """Waveform class for general waveform"""

    waveform_times = properties.Array(
        "Time for input currents", dtype=float
    )

    waveform_current = properties.Array(
        "Input currents", dtype=float
    )

    base_frequency = properties.Float(
        "Base frequency (Hz)", default=30.
    )

    def __init__(self, **kwargs):
        super(StepoffWaveform, self).__init__(**kwargs)

    @property
    def wave_type(self):
        return "general"

    @property
    def period(self):
        return 1./self.base_frequency

    @property
    def pulse_period(self):
        Tp = (
            self.waveform_times.max() -
            self.waveform_times.min()
        )
        return Tp


class DualWaveform(GeneralWaveform):
    """Waveform class supporting primary and dual moment waveforms"""

    dual_waveform_times = properties.Array(
        "Time for input currents (dual moment)", dtype=float
    )

    dual_waveform_current = properties.Array(
        "Input currents (dual moment)", dtype=float
    )

    dual_base_frequency = properties.Float(
        "Base frequency for the dual moment", default=30.
    )

    def __init__(self, **kwargs):
        super(DualWaveform, self).__init__(**kwargs)

    @property
    def wave_type(self):
        return "dual"

    @property
    def dual_period(self):
        return 1./self.dual_base_frequency

    @property
    def dual_pulse_period(self):
        Tp = (
            self.dual_waveform_times.max() -
            self.dual_waveform_times.min()
        )
        return Tp

############################################################
#               SIMPLE WAVEFORM CLASSES
############################################################

class RectangularWaveform(GeneralWaveform):
    """Rectangular waveform"""

    _waveform_current = None

    def __init__(self, waveform_times, start_time, end_time, peak_current_amplitude=1., **kwargs):
        super(RectangularWaveform, self).__init__(**kwargs)

        self.waveform_times = waveform_times
        self.start_time = start_time
        self.end_time = end_time
        self.peak_current_amplitude = peak_current_amplitude

    @property
    def waveform_current(self):

        if self._waveform_current is None:
            # temp = np.zeros(self.waveform_times.size)
            # temp[(self.waveform_times>self.start_time) & (self.waveform_times<self.end_time)] = self.peak_current_amplitude
            # self._waveform_current = 

            self._waveform_current = rectangular_waveform_current(
                self.waveform_times, self.start_time, self.end_time, self.peak_current_amplitude
            )

        return self._waveform_current
        
class TriangleWaveform(GeneralWaveform):
    """
        Triangular Waveform
        * time: 1D array for time
        * on_time: start of on-time
        * peak_time: peak time
        * off_time: off-time
    """

    _waveform_current = None

    def __init__(self, waveform_times, start_time, peak_time, end_time, peak_current_amplitude=1., **kwargs):
        super(TriangleWaveform, self).__init__(**kwargs)

        self.waveform_times = waveform_times
        self.start_time = start_time
        self.peak_time = peak_time
        self.end_time = end_time
        self.peak_current_amplitude = peak_current_amplitude

    @property
    def waveform_current(self):

        if self._waveform_current is None:
            # t = self.waveform_times
            # temp = np.zeros(t.size)
            # k = (t>=self.start_time) & (t<=self.peak_time)
            # temp[k] = (t[k] - self.start_time) * self.peak_current_amplitude / (self.peak_time - self.start_time) 
            # k = (t>=self.peak_time) & (t<=self.end_time)
            # temp[k] = self.peak_current_amplitude * (1 - (t[k] - self.peak_time) / (self.end_time - self.peak_time))
            # self._waveform_current = temp

            self._waveform_current = triangular_waveform_current(
                self.waveform_times, self.start_time, self.peak_time, self.end_time, self.peak_current_amplitude
            )

        return self._waveform_current


class VTEMCustomWaveform(GeneralWaveform):

    _waveform_current = None

    def __init__(self, waveform_times, start_time, peak_time, end_time, decay_constant, peak_current_amplitude=1., **kwargs):
        super(VTEMCustomWaveform, self).__init__(**kwargs)

        self.waveform_times = waveform_times
        self.start_time = start_time
        self.peak_time = peak_time
        self.end_time = end_time
        self.decay_constant = decay_constant
        self.peak_current_amplitude = peak_current_amplitude
    
    @property
    def waveform_current(self):

        if self._waveform_current is None:
            # t = self.waveform_times
            # out = np.zeros(t.size)

            # k = (t>=self.start_time) & (t<=self.peak_time)
            # out[k] = (
            #     self.peak_current_amplitude *
            #     (1 - np.exp(-self.decay_constant*(t[k] - self.start_time))) / 
            #     (1 - np.exp(-self.decay_constant*(self.peak_time - self.start_time)))
            # )

            # k = (t>=self.peak_time) & (t<=self.end_time)
            # out[k] = self.peak_current_amplitude * (1 - (t[k] - self.peak_time) / (self.end_time - self.peak_time))
            
            # return out

            self._waveform_current = vtem_waveform_current(
                self.waveform_times, self.start_time, peak_time, self.end_time, self.decay_constant, self.peak_current_amplitude
            )

        return self._waveform_current


############################################################
#              WAVEFORM CLASSES FOR KNOWN SYSTEMS
############################################################

class VTEMPlusWaveform(GeneralWaveform):

    off_time = None
    peak_current_amplitude = None

    def __init__(self, **kwargs):
        super(VTEMPlusWaveform, self).__init__(**kwargs)

        self.off_time = kwargs.get('off_time', 0.00734375)
        self.peak_current_amplitude = kwargs.get('peak_current_amplitude', 1.)

    @property
    def base_frequency(self):
        return 25.

    @property
    def waveform_times(self):
        return vtem_plus_waveform_times(self.off_time)

    @property
    def waveform_current(self):
        return vtem_plus_waveform_current(self.peak_current_amplitude)


class Skytem2015LowMomentWaveform(GeneralWaveform):
    """
        SkyTEM High moment (HM) current waveform
    """

    off_time = None
    peak_current_amplitude = None

    def __init__(self, **kwargs):

        super(Skytem2015LowMomentWaveform, self).__init__(**kwargs)

        self.off_time = kwargs.get('off_time', 9.4274e-006)
        self.peak_current_amplitude = kwargs.get('peak_current_amplitude', 8.3)


    # Define the high moment
    @property
    def base_frequency(self):
        return 210.
    
    @property
    def waveform_times(self):
        return skytem_2015_LM_waveform_times(self.off_time)

    @property
    def waveform_current(self):
        return skytem_2015_LM_waveform_current(self.peak_current_amplitude)

class Skytem2015HighMomentWaveform(GeneralWaveform):
    """
        SkyTEM High moment (HM) current waveform
    """

    off_time = None
    peak_current_amplitude = None

    def __init__(self, **kwargs):
        
        super(Skytem2015HighMomentWaveform, self).__init__(**kwargs)

        self.off_time = kwargs.get('off_time', 1.96368E-04)
        self.peak_current_amplitude = kwargs.get('peak_current_amplitude', 122.5)

    # Define the high moment
    @property
    def base_frequency(self):
        return 30.

    @property
    def waveform_times(self):
        return skytem_2015_HM_waveform_times(self.off_time)

    @property
    def waveform_current(self):
        return skytem_2015_HM_waveform_current(self.peak_current_amplitude)


class Skytem2015Waveform(DualWaveform):
    """
        Full SkyTEM 2015 waveform. Includes low monent and high moment waveforms.
    """

    off_time = None
    peak_current_amplitude = None
    dual_off_time = None
    dual_peak_current_amplitude = None

    def __init__(self, **kwargs):

        super(Skytem2015Waveform, self).__init__(**kwargs)

        self.off_time = kwargs.get('off_time', 9.4274e-006)
        self.peak_current_amplitude = kwargs.get('peak_current_amplitude', 8.3)
        self.dual_off_time = kwargs.get('dual_off_time', 1.96368E-04)
        self.dual_peak_current_amplitude = kwargs.get('dual_peak_current_amplitude', 122.5)
    
    @property
    def base_frequency(self):
        return 210.

    @property
    def dual_base_frequency(self):
        return 30.
    
    @property
    def waveform_times(self):
        return skytem_2015_LM_waveform_times(self.off_time)

    @property
    def waveform_current(self):
        return skytem_2015_LM_waveform_current(self.peak_current_amplitude)

    @property
    def dual_waveform_times(self):
        return skytem_2015_HM_waveform_times(self.dual_off_time)

    @property
    def dual_waveform_current(self):
        return skytem_2015_HM_waveform_current(self.dual_peak_current_amplitude)

    
