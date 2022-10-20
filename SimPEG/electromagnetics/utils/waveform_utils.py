import numpy as np
from scipy.constants import mu_0, epsilon_0
from scipy import integrate

# useful params
def omega(frequency):
    """Compute angular frequency from frequency

    Parameters
    ----------
    frequency : float or numpy.ndarray
        Frequencies in Hz

    Returns
    -------
    float or numpy.ndarray
        Angular frequencies in rad/s
    """
    return 2.0 * np.pi * frequency


def k(frequency, sigma, mu=mu_0, eps=epsilon_0):
    r"""Wavenumber for EM waves in homogeneous media

    See eq 1.47 - 1.49 in Ward and Hohmann

    Parameters
    ----------
    frequency : float or numpy.ndarray
        Frequencies in Hz
    sigma : float
        Electrical conductivity in S/m
    mu : float, default: :math:`4\pi \;\times\; 10^{-7}` H/m
        Magnetic permeability in H/m
    eps : float, default: 8.8541878128 :math:`\times \; 1-^{-12}` F/m
        Dielectric permittivity in F/m

    Returns
    -------
    complex or numpy.ndarray
        Wavenumbers at all input frequencies
    """
    w = omega(frequency)
    alp = w * np.sqrt(mu * eps / 2 * (np.sqrt(1.0 + (sigma / (eps * w)) ** 2) + 1))
    beta = w * np.sqrt(mu * eps / 2 * (np.sqrt(1.0 + (sigma / (eps * w)) ** 2) - 1))
    return alp - 1j * beta


def VTEMFun(time, ta, tb, a):
    """VTEM waveform function

    Parameters
    ----------
    time : numpy.ndarray
        Times vector
    ta : float
        Time at peak exponential
    tb : float
        Start of off-time

    Returns
    -------
    (n_time) numpy.ndarray
        The waveform evaluated at all input times
    """
    out = np.zeros(time.size)
    out[time <= ta] = (1 - np.exp(-a * time[time <= ta] / ta)) / (1 - np.exp(-a))
    out[(time > ta) & (time < tb)] = (
        -1 / (tb - ta) * (time[(time > ta) & (time < tb)] - tb)
    )
    return out


def convolve_with_waveform(func, waveform, times, fargs=None, fkwargs=None):
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
    fargs : list, optional
        extra arguments given to `func`
    fkwargs : dict, optional
        keyword arguments given to `func`

    Returns
    -------
    numpy.ndarray
        the convolution evaluate at the given times
    """
    try:
        t_nodes = waveform.time_nodes
    except AttributeError:
        raise TypeError(f"Unsupported waveform type of {type(waveform)}")

    if fargs is None:
        fargs = []
    if fkwargs is None:
        fkwargs = {}

    n_int = len(t_nodes) - 1
    out = np.zeros_like(times, dtype=float)
    for it, t in enumerate(times):

        def integral(quad_time):
            wave_eval = waveform.eval_deriv(t - quad_time)
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
