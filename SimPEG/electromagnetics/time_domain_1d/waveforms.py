import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import mu_0

def TriangleFun(time, ta, tb):
    """
        Triangular Waveform
        * time: 1D array for time
        * ta: time at peak
        * tb: time at step-off
    """
    out = np.zeros(time.size)
    out[time<=ta] = 1/ta*time[time<=ta]
    out[(time>ta)&(time<tb)] = -1/(tb-ta)*(time[(time>ta)&(time<tb)]-tb)
    return out

def TriangleFunDeriv(time, ta, tb):
    """
        Derivative of Triangular Waveform
    """
    out = np.zeros(time.size)
    out[time<=ta] = 1/ta
    out[(time>ta)&(time<tb)] = -1/(tb-ta)
    return out

def SineFun(time, ta):
    """
        Sine Waveform
        * time: 1D array for time
        * ta: Pulse Period
    """
    out = np.zeros(time.size)
    out[time<=ta] = np.sin(1./ta*np.pi*time[time<=ta])

    return out

def SineFunDeriv(time, ta):
    """
        Derivative of Sine Waveform
    """
    out = np.zeros(time.size)
    out[time<=ta] = 1./ta*np.pi*np.cos(1./ta*np.pi*time[time<=ta])
    return out


def VTEMFun(time, ta, tb, a):
    """
        VTEM Waveform
        * time: 1D array for time
        * ta: time at peak of exponential part
        * tb: time at step-off
    """
    out = np.zeros(time.size)
    out[time<=ta] = (1-np.exp(-a*time[time<=ta]/ta))/(1-np.exp(-a))
    out[(time>ta)&(time<tb)] = -1/(tb-ta)*(time[(time>ta)&(time<tb)]-tb)
    return out

def CausalConv(array1, array2, time):
    """
        Evaluate convolution for two causal functions.
        Input

        * array1: array for \\\\(\\\\ f_1(t)\\\\)
        * array2: array for \\\\(\\\\ f_2(t)\\\\)
        * time: array for time

        .. math::

            Out(t) = \int_{0}^{t} f_1(a) f_2(t-a) da

    """

    if array1.shape == array2.shape == time.shape:
        out = np.convolve(array1, array2)
        # print time[1]-time[0]
        return out[0:np.size(time)]*(time[1]-time[0])
    else:
        print ("Give me same size of 1D arrays!!")


def RectFun(time, ta, tb):
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
    out = np.zeros(time.size)
    out[time<=ta] = 1
    out[(time>ta)&(time<tb)] = -1
    return out


def CenDiff(f, tin):
    """
        Evaluating central difference of given array (f)
        and provide funtion handle for interpolation
    """
    dfdt = mu_0*np.diff(f, n=1)/np.diff(tin, n=1)
    tm = np.diff(tin, n=1)*0.5 + tin[:-1]
    Diffun = interp1d(tm, dfdt)
    return Diffun
