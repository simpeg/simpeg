from __future__ import division
import numpy as np
from scipy.constants import mu_0, pi
from scipy.special import erf
import matplotlib.pyplot as plt
from SimPEG import Utils

def hzAnalyticDipoleF(r, freq, sigma, secondary=True):
    """
    4.56 in Ward and Hohmann

    .. plot::

        import matplotlib.pyplot as plt
        import simpegEM as EM
        freq = np.logspace(-1, 6, 61)
        test = EM.Analytics.FDEM.hzAnalyticDipoleF(100, freq, 0.001, secondary=False)
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

def AnalyticMagDipoleWholeSpace(XYZ, srcLoc, sig, f, m=1., orientation='X'):
    """
    Analytical solution for a dipole in a whole-space.

    Equation 2.57 of Ward and Hohmann

    TODOs:
        - set it up to instead take a mesh & survey
        - add E-fields
        - handle multiple frequencies
        - add divide by zero safety


    .. plot::

        import simpegEM as EM
        import matplotlib.pyplot as plt
        freqs = np.logspace(-2,5,100)
        Bx, By, Bz = EM.Analytics.FDEM.AnalyticMagDipoleWholeSpace([0,100,0], [0,0,0], 1e-2, freqs, m=1, orientation='Z')
        plt.loglog(freqs, np.abs(Bz.real)/mu_0, 'b')
        plt.loglog(freqs, np.abs(Bz.imag)/mu_0, 'r')
        plt.legend(('real','imag'))
        plt.show()


    """

    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)

    dx = XYZ[:,0]-srcLoc[0]
    dy = XYZ[:,1]-srcLoc[1]
    dz = XYZ[:,2]-srcLoc[2]

    r  = np.sqrt( dx**2. + dy**2. + dz**2.)
    k  = np.sqrt( -1j*2.*np.pi*f*mu_0*sig )
    kr = k*r

    front = m / (4.*pi * r**3.) * np.exp(-1j*kr)
    mid   = -kr**2. + 3.*1j*kr + 3.

    if orientation.upper() == 'X':
        Hx = front*( (dx/r)**2. * mid + (kr**2. - 1j*kr - 1.) )
        Hy = front*( (dx*dy/r**2.) * mid )
        Hz = front*( (dx*dz/r**2.) * mid )

    elif orientation.upper() == 'Y':
        Hx = front*( (dy*dx/r**2.) * mid )
        Hy = front*( (dy/r)**2. * mid + (kr**2. - 1j*kr - 1.) )
        Hz = front*( (dy*dz/r**2.) * mid )

    elif orientation.upper() == 'Z':
        Hx = front*( (dx*dz/r**2.) * mid )
        Hy = front*( (dy*dz/r**2.) * mid )
        Hz = front*( (dz/r)**2. * mid + (kr**2. - 1j*kr - 1.) )

    Bx = mu_0*Hx
    By = mu_0*Hy
    Bz = mu_0*Hz
    return Bx, By, Bz

