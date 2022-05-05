import numpy as np
from scipy.constants import mu_0, epsilon_0

# useful params
def omega(freq):
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


def TriangleFun(time, ta, tb):
    """Triangular waveform function

    Parameters
    ----------
    time : numpy.ndarray
        Times vector
    ta : float
        Peak time
    tb : float
        Start of off-time

    Returns
    -------
    (n_time) numpy.ndarray
        The waveform evaluated at all input times
    """
    out = np.zeros(time.size)
    out[time <= ta] = 1 / ta * time[time <= ta]
    out[(time > ta) & (time < tb)] = (
        -1 / (tb - ta) * (time[(time > ta) & (time < tb)] - tb)
    )
    return out


def TriangleFunDeriv(time, ta, tb):
    """Derivative of triangular waveform function wrt time

    Parameters
    ----------
    time : numpy.ndarray
        Times vector
    ta : float
        Peak time
    tb : float
        Start of off-time

    Returns
    -------
    (n_time) numpy.ndarray
        Derivative wrt to time at all input times
    """
    out = np.zeros(time.size)
    out[time <= ta] = 1 / ta
    out[(time > ta) & (time < tb)] = -1 / (tb - ta)
    return out


def SineFun(time, ta):
    """Sine waveform function

    Parameters
    ----------
    time : numpy.ndarray
        Times vector
    ta : float
        Pulse period

    Returns
    -------
    (n_time) numpy.ndarray
        The waveform evaluated at all input times
    """
    out = np.zeros(time.size)
    out[time <= ta] = np.sin(1.0 / ta * np.pi * time[time <= ta])

    return out


def SineFunDeriv(time, ta):
    """Derivative of sine waveform function

    Parameters
    ----------
    time : numpy.ndarray
        Times vector
    ta : float
        Pulse period

    Returns
    -------
    (n_time) numpy.ndarray
        The waveform evaluated at all input times
    """
    out = np.zeros(time.size)
    out[time <= ta] = 1.0 / ta * np.pi * np.cos(1.0 / ta * np.pi * time[time <= ta])
    return out


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
