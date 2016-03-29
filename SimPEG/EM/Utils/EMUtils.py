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


