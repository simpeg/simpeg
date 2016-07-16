from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from past.utils import old_div
import numpy as np
from scipy.constants import mu_0, epsilon_0

# useful params
def omega(freq):
    """Angular frequency, omega"""
    return 2.*np.pi*freq

def k(freq, sigma, mu=mu_0, eps=epsilon_0):
    """ Eq 1.47 - 1.49 in Ward and Hohmann """
    w = omega(freq)
    alp  = w * np.sqrt( mu*eps/2 * ( np.sqrt(1. + (old_div(sigma, (eps*w)))**2 ) + 1) ) 
    beta = w * np.sqrt( mu*eps/2 * ( np.sqrt(1. + (old_div(sigma, (eps*w)))**2 ) - 1) ) 
    return alp - 1j*beta


