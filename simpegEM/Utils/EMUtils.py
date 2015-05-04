import numpy as np
from scipy.constants import mu_0, epsilon_0

def omega(freq):
    """Angular frequency, omega"""
    return 2.*np.pi*freq

def k(freq, sigma, mu=mu_0, eps=epsilon_0):
	w = omega(freq)
	return np.sqrt(mu * eps * w**2 - 1j * w* mu * sigma)