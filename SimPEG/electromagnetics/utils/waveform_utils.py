import numpy as np
from scipy.constants import mu_0, epsilon_0
from scipy import integrate

# useful params
def omega(freq):
    """Angular frequency, omega"""
    return 2.0 * np.pi * freq


def k(freq, sigma, mu=mu_0, eps=epsilon_0):
    """Eq 1.47 - 1.49 in Ward and Hohmann"""
    w = omega(freq)
    alp = w * np.sqrt(mu * eps / 2 * (np.sqrt(1.0 + (sigma / (eps * w)) ** 2) + 1))
    beta = w * np.sqrt(mu * eps / 2 * (np.sqrt(1.0 + (sigma / (eps * w)) ** 2) - 1))
    return alp - 1j * beta


def TriangleFun(time, ta, tb):
    """
    Triangular Waveform
    * time: 1D array for time
    * ta: time at peak
    * tb: time at step-off
    """
    out = np.zeros(time.size)
    out[time <= ta] = 1 / ta * time[time <= ta]
    out[(time > ta) & (time < tb)] = (
        -1 / (tb - ta) * (time[(time > ta) & (time < tb)] - tb)
    )
    return out


def TriangleFunDeriv(time, ta, tb):
    """
    Derivative of Triangular Waveform
    """
    out = np.zeros(time.size)
    out[time <= ta] = 1 / ta
    out[(time > ta) & (time < tb)] = -1 / (tb - ta)
    return out


def SineFun(time, ta):
    """
    Sine Waveform
    * time: 1D array for time
    * ta: Pulse Period
    """
    out = np.zeros(time.size)
    out[time <= ta] = np.sin(1.0 / ta * np.pi * time[time <= ta])

    return out


def SineFunDeriv(time, ta):
    """
    Derivative of Sine Waveform
    """
    out = np.zeros(time.size)
    out[time <= ta] = 1.0 / ta * np.pi * np.cos(1.0 / ta * np.pi * time[time <= ta])
    return out


def VTEMFun(time, ta, tb, a):
    """
    VTEM Waveform
    * time: 1D array for time
    * ta: time at peak of exponential part
    * tb: time at step-off
    """
    out = np.zeros(time.size)
    out[time <= ta] = (1 - np.exp(-a * time[time <= ta] / ta)) / (1 - np.exp(-a))
    out[(time > ta) & (time < tb)] = (
        -1 / (tb - ta) * (time[(time > ta) & (time < tb)] - tb)
    )
    return out


def convolve_with_waveform(func, waveform, times, fargs=[], fkwargs={}):
    """convolves the function with the given waveform

    This convolves the given function with the waveform and evaluates it at the given
    times. This uses a high order Gaussian-Legendre quadrature to evaluate, and could
    likely be slow.

    Parameters
    ----------
    func : callable
        function of `t` that should be convolved
    waveform : SimPEG.electromagnetics.time_domain.waveforms.BaseWaveform
    times : array_like
    fargs : list
        extra arguments given to `func`
    fkwargs : dict
        keyword arguments given to `func`

    Returns
    -------
    np.ndarray
        the convolution evaluate at the given times
    """
    try:
        t_nodes = waveform.time_nodes
    except AttributeError:
        raise TypeError(f"Unsupported waveform type of {type(waveform)}")

    n_int = len(t_nodes) - 1
    out = np.zeros_like(times, dtype=float)
    for it, t in enumerate(times):

        def integral(quad_time):
            wave_eval = waveform.evalDeriv(t - quad_time)
            return wave_eval * func(quad_time, *fargs, **fkwargs)

        for i in range(n_int):
            b = t - t_nodes[i]
            a = t - t_nodes[i + 1]
            # just do not evaluate the integral at negative times...
            a = np.maximum(a, 0.0)
            b = np.maximum(b, 0.0)
            val, _ = integrate.quadrature(integral, a, b, tol=0.0, maxiter=500)
            out[it] -= val
    return out
