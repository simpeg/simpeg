import numpy as np
from scipy.constants import mu_0, pi
from scipy.special import erf

def hzAnalyticDipoleF(r, freq, sigma):
    """
    4.56 in Ward and Hohmann
    """
    r = np.abs(r)
    k = np.sqrt(-1j*2.*np.pi*freq*mu_0*sigma)

    m = 1
    front = m / (2. * np.pi * (k**2) * (r**5) )
    back = 9 - ( 9 + 9j * k * r - 4 * (k**2) * (r**2) - 1j * (k**3) * (r**3)) * np.exp(-1j*k*r)
    hz = front*back
    hp =-1/(4*np.pi*r**3)
    return hz-hp
