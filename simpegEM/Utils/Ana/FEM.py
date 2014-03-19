import numpy as np
from scipy.constants import mu_0, pi
from scipy.special import erf

def hzAnalyticDipoleF(r, freq, sigma, secondary=True):
    """
    4.56 in Ward and Hohmann

    .. plot::

        import matplotlib.pyplot as plt
        import simpegEM as EM
        freq = np.logspace(-1, 6, 61)
        test = EM.Utils.Ana.FEM.hzAnalyticDipoleF(100, freq, 0.001, secondary=False)
        plt.loglog(freq, abs(test.real))
        plt.loglog(freq, abs(test.imag))
        plt.title('Response at $r$=100m')
        plt.xlabel('Frequency')
        plt.ylabel('Response')
        plt.legend(('real','imag'))
        plt.show()

    """
    r = np.abs(r)
    k = np.sqrt(-1j*2.*np.pi*freq*mu_0*sigma)

    m = 1
    front = m / (2. * np.pi * (k**2) * (r**5) )
    back = 9 - ( 9 + 9j * k * r - 4 * (k**2) * (r**2) - 1j * (k**3) * (r**3)) * np.exp(-1j*k*r)
    hz = front*back

    if secondary:
        hp =-1/(4*np.pi*r**3)
        return hz-hp

    return hz


