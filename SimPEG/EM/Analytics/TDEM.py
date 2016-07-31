from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import numpy as np
from scipy.constants import mu_0, pi
from scipy.special import erf

def hzAnalyticDipoleT(r, t, sigma):
    theta = np.sqrt((sigma*mu_0)/(4*t))
    tr = theta*r
    etr = erf(tr)
    t1 = (9/(2*tr**2) - 1)*etr
    t2 = (1/np.sqrt(pi))*(9/tr + 4*tr)*np.exp(-tr**2)
    hz = (t1 - t2)/(4*pi*r**3)
    return hz
