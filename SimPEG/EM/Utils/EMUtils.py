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

# # Constitutive relations
# def e_from_j(prob,j):
#     eqLocs = prob._eqLocs
#     if eqLocs is 'FE':
#         MSigmaI = prob.MeSigmaI
#     elif eqLocs is 'EF':
#         MSigmaI = prob.MfRho
#     return MSigmaI*j

# def j_from_e(prob,e):
#     eqLocs = prob._eqLocs
#     if eqLocs is 'FE':
#         MSigma = prob.MeSigma
#     elif eqLocs is 'EF':
#         MSigma = prob.MfRhoI
#     return MSigma*e

# def b_from_h(prob,h):
#     eqLocs = prob._eqLocs
#     if eqLocs is 'FE':
#         MMu = prob.MfMuiI
#     elif eqLocs is 'EF':
#         MMu = prob.MeMu
#     return MMu*h

# def h_from_b(prob,b):
#     eqLocs = prob._eqLocs
#     if eqLocs is 'FE':
#         MMuI = prob.MfMui
#     elif eqLocs is 'EF':
#         MMuI = prob.MeMuI
#     return MMuI*b 


