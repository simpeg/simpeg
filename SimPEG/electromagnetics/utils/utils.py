import numpy as np
from scipy.constants import mu_0, epsilon_0

# useful params
def omega(freq):
    """Angular frequency, omega"""
    return 2.*np.pi*freq

def k(freq, sigma, mu=mu_0, eps=epsilon_0):
    """ Eq 1.47 - 1.49 in Ward and Hohmann """
    w = omega(freq)
    alp  = w * np.sqrt( mu*eps/2 * ( np.sqrt(1. + (sigma / (eps*w))**2 ) + 1) )
    beta = w * np.sqrt( mu*eps/2 * ( np.sqrt(1. + (sigma / (eps*w))**2 ) - 1) )
    return alp - 1j*beta

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
