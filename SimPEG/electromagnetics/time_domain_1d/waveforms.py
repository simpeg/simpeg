import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import mu_0
import properties

class BaseWaveformEM1DTM(properties.HasProperties):
    """Base waveform class for EM1DTM simulations."""


    wave_type = properties.StringChoice(
        "Waveform type",
        default="stepoff",
        choices=["stepoff", "general", "dual"]
    )

    # moment_type = properties.StringChoice(
    #     "Source moment type",
    #     default="single",
    #     choices=["single", "dual"]
    # )

    n_pulse = properties.Integer(
        "The number of pulses",
        default=1
    )

    base_frequency = properties.Float(
        "Base frequency (Hz)"
    )

    waveform_times = properties.Array(
        "Time for input currents", dtype=float
    )

#    waveform_currents = properties.Array(
#        "Input currents", dtype=float
#    )
    
    _waveform_currents = None

    use_lowpass_filter = properties.Bool(
        "Switch for low pass filter", default=False
    )

    high_cut_frequency = properties.Float(
        "High cut frequency for low pass filter (Hz)",
        default=210*1e3
    )


    # ------------- For dual moment ------------- #

    dual_waveform_times = properties.Array(
        "Time for input currents (dual moment)", dtype=float
    )

    dual_waveform_currents = properties.Array(
        "Input currents (dual moment)", dtype=float
    )

    dual_base_frequency = properties.Float(
        "Base frequency for the dual moment"
    )
    
    def __init__(self, **kwargs):
        super(BaseWaveformEM1DTM, self).__init__(**kwargs)


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

    # ------------- For dual moment ------------- #
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

class StepoffWaveform(BaseWaveformEM1DTM):

    def __init__(self, **kwargs):
        super(StepoffWaveform, self).__init__(wave_type='stepoff', **kwargs)


class SquareWaveform(BaseWaveformEM1DTM):
    """

        Rectangular Waveform

        * time: 1D array for time
        * ta: time for transition from (+) to (-)
        * tb: time at step-off

        .. math::

            I(t) = 1, 0 < t \\le t_a

            I(t) = -1, t_a < t < t_b

            I(t) = 0, t \\le t_a \\ \\text{or}  \\ t \\ge t_b

    """
    def __init__(self, start_time, end_time, peak_current_amplitude=1., **kwargs):
        
        super(SquareWaveform, self).__init__(
            wave_type='general', **kwargs
        )

        self.start_time = start_time
        self.end_time = end_time
        self.peak_current_amplitude = peak_current_amplitude

    @property
    def waveform_currents(self):

        if getattr(self, "_waveform_currents", None) is None:
            temp = np.zeros(self.waveform_times.size)
            temp[(self.waveform_times>self.start_time) & (self.waveform_times<self.end_time)] = self.peak_current_amplitude
            self._waveform_currents = temp

        return self._waveform_currents
        
class TriangleWaveform(BaseWaveformEM1DTM):
    """
        Triangular Waveform
        * time: 1D array for time
        * on_time: start of on-time
        * peak_time: peak time
        * off_time: off-time
    """

    def __init__(self, start_time, peak_time, end_time, peak_current_amplitude=1., **kwargs):
        super(TriangleWaveform, self).__init__(wave_type='general', **kwargs)

        self.start_time = start_time
        self.peak_time = peak_time
        self.end_time = end_time
        self.peak_current_amplitude = peak_current_amplitude

    @property
    def waveform_currents(self):

        if getattr(self, "_waveform_currents", None) is None:
            t = self.waveform_times
            temp = np.zeros(t.size)
            k = (t>=self.start_time) & (t<=self.peak_time)
            temp[k] = (t[k] - self.start_time) * self.peak_current_amplitude / (self.peak_time - self.start_time) 
            k = (t>=self.peak_time) & (t<=self.end_time)
            temp[k] = self.peak_current_amplitude * (1 - (t[k] - self.peak_time) / (self.end_time - self.peak_time))

            self._waveform_currents = temp

        return self._waveform_currents


# def TriangleFunDeriv(time, ta, tb):
#     """
#         Derivative of Triangular Waveform
#     """
#     out = np.zeros(time.size)
#     out[time<=ta] = 1/ta
#     out[(time>ta)&(time<tb)] = -1/(tb-ta)
#     return out

# def SineFun(time, ta):
#     """
#         Sine Waveform
#         * time: 1D array for time
#         * ta: Pulse Period
#     """
#     out = np.zeros(time.size)
#     out[time<=ta] = np.sin(1./ta*np.pi*time[time<=ta])

#     return out

# def SineFunDeriv(time, ta):
#     """
#         Derivative of Sine Waveform
#     """
#     out = np.zeros(time.size)
#     out[time<=ta] = 1./ta*np.pi*np.cos(1./ta*np.pi*time[time<=ta])
#     return out


# def VTEMFun(time, ta, tb, a):
#     """
#         VTEM Waveform
#         * time: 1D array for time
#         * ta: time at peak of exponential part
#         * tb: time at step-off
#     """
#     out = np.zeros(time.size)
#     out[time<=ta] = (1-np.exp(-a*time[time<=ta]/ta))/(1-np.exp(-a))
#     out[(time>ta)&(time<tb)] = -1/(tb-ta)*(time[(time>ta)&(time<tb)]-tb)
#     return out

# def CausalConv(array1, array2, time):
#     """
#         Evaluate convolution for two causal functions.
#         Input

#         * array1: array for \\\\(\\\\ f_1(t)\\\\)
#         * array2: array for \\\\(\\\\ f_2(t)\\\\)
#         * time: array for time

#         .. math::

#             Out(t) = \int_{0}^{t} f_1(a) f_2(t-a) da

#     """

#     if array1.shape == array2.shape == time.shape:
#         out = np.convolve(array1, array2)
#         # print time[1]-time[0]
#         return out[0:np.size(time)]*(time[1]-time[0])
#     else:
#         print ("Give me same size of 1D arrays!!")




# def CenDiff(f, tin):
#     """
#         Evaluating central difference of given array (f)
#         and provide funtion handle for interpolation
#     """
#     dfdt = mu_0*np.diff(f, n=1)/np.diff(tin, n=1)
#     tm = np.diff(tin, n=1)*0.5 + tin[:-1]
#     Diffun = interp1d(tm, dfdt)
#     return Diffun





class SkytemHM2015Waveform(BaseWaveformEM1DTM):
    """
        SkyTEM High moment (HM) current waveform
    """

    def __init__(self, peak_time=0., peak_current_amplitude=122.5, **kwargs):
        
        super(SkytemHM2015Waveform, self).__init__(wave_type="general", **kwargs)

        # Define the high moment
        self.base_frequency = 30.
        self.waveform_times = np.array([
            -2.06670E-02,
            -2.05770E-02,
            -2.04670E-02,
            -1.66670E-02,
            -1.64726E-02,
            -1.64720E-02,
            -1.64706E-02,
            -4.00000E-03,
            -3.91000E-03,
            -3.80000E-03,
            0.00000E+00,
            1.94367E-04,
            1.95038E-04,
            1.96368E-04
        ]) - peak_time

        self.waveform_currents = np.array([
            0.00000E+00,
            -5.30000E-01,
            -9.73000E-01,
            -1.00000E+00,
            -2.81610E-03,
            -1.44356E-03,
            0.00000E+00,
            0.00000E+00,
            5.30000E-01,
            9.73000E-01,
            1.00000E+00,
            2.81610E-03,
            1.44356E-03,
            0.00000E+00
        ]) * peak_current_amplitude


class SkytemLM2015Waveform(BaseWaveformEM1DTM):
    """
        SkyTEM High moment (HM) current waveform
    """

    def __init__(self, peak_time=0., peak_current_amplitude=8.3, **kwargs):

        super(SkytemLM2015Waveform, self).__init__(wave_type="general", **kwargs)

        # Define the high moment
        self.base_frequency = 210.
        self.waveform_times = np.array([
            -3.1810e-003,
            -3.1100e-003,
            -2.7860e-003,
            -2.5334e-003,
            -2.3820e-003,
            -2.3810e-003,
            -2.3798e-003,
            -2.3779e-003,
            -2.3762e-003,
            -2.3749e-003,
            -2.3733e-003,
            -2.3719e-003,
            -2.3716e-003,
            -8.0000e-004,
            -7.2902e-004,
            -4.0497e-004,
            -1.5238e-004,
            -1.0000e-006,
            0,
            1.1535e-006,
            3.0943e-006,
            4.7797e-006,
            6.1076e-006,
            7.7420e-006,
            9.0699e-006,
            9.4274e-006,
        ]) - peak_time

        self.waveform_currents = np.array([
            0,
            -1.0078e-001,
            -4.5234e-001,
            -7.6328e-001,
            -1.0000e+000,
            -1.0000e+000,
            -8.6353e-001,
            -3.4002e-001,
            -1.1033e-001,
            -4.4709e-002,
            -1.3388e-002,
            -4.4389e-003,
            0,
            0,
            1.0078e-001,
            4.5234e-001,
            7.6328e-001,
            1.0000e+000,
            1.0000e+000,
            8.6353e-001,
            3.4002e-001,
            1.1033e-001,
            4.4709e-002,
            1.3388e-002,
            4.4389e-003,
            0
        ]) * peak_current_amplitude

    
