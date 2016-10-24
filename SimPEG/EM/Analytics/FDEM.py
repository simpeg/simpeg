from __future__ import division
import numpy as np
from scipy.constants import mu_0, pi
from scipy.special import erf
from SimPEG import Utils


def hzAnalyticDipoleF(r, freq, sigma, secondary=True, mu=mu_0):
    """
    4.56 in Ward and Hohmann

    .. plot::

        import matplotlib.pyplot as plt
        from SimPEG import EM
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
    k = np.sqrt(-1j*2.*np.pi*freq*mu*sigma)

    m = 1
    front = m / (2. * np.pi * (k**2) * (r**5) )
    back = 9 - ( 9 + 9j * k * r - 4 * (k**2) * (r**2) - 1j * (k**3) * (r**3)) * np.exp(-1j*k*r)
    hz = front*back

    if secondary:
        hp =-1/(4*np.pi*r**3)
        hz = hz-hp

    if hz.ndim == 1:
        hz = Utils.mkvc(hz,2)

    return hz

def MagneticDipoleWholeSpace(XYZ, srcLoc, sig, f, moment=1., orientation='X', mu = mu_0):
    """
    Analytical solution for a dipole in a whole-space.

    Equation 2.57 of Ward and Hohmann

    TODOs:
        - set it up to instead take a mesh & survey
        - add E-fields
        - handle multiple frequencies
        - add divide by zero safety


    .. plot::

        from SimPEG import EM
        import matplotlib.pyplot as plt
        from scipy.constants import mu_0
        freqs = np.logspace(-2,5,100)
        Bx, By, Bz = EM.Analytics.FDEM.MagneticDipoleWholeSpace([0,100,0], [0,0,0], 1e-2, freqs, moment=1, orientation='Z')
        plt.loglog(freqs, np.abs(Bz.real)/mu_0, 'b')
        plt.loglog(freqs, np.abs(Bz.imag)/mu_0, 'r')
        plt.legend(('real','imag'))
        plt.show()


    """

    if not isinstance(orientation, str):
        if np.allclose(orientation, np.r_[1., 0., 0.]):
            orientation = 'X'
        elif np.allclose(orientation, np.r_[0., 1., 0.]):
            orientation = 'Y'
        elif np.allclose(orientation, np.r_[0., 0., 1.]):
            orientation = 'Z'
        else:
            raise NotImplementedError('arbitrary orientations not implemented')

    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)

    dx = XYZ[:,0]-srcLoc[0]
    dy = XYZ[:,1]-srcLoc[1]
    dz = XYZ[:,2]-srcLoc[2]

    r  = np.sqrt( dx**2. + dy**2. + dz**2.)
    k  = np.sqrt( -1j*2.*np.pi*f*mu*sig )
    kr = k*r

    front = moment / (4.*pi * r**3.) * np.exp(-1j*kr)
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

    Bx = mu*Hx
    By = mu*Hy
    Bz = mu*Hz

    if Bx.ndim is 1:
        Bx = Utils.mkvc(Bx,2)

    if By.ndim is 1:
        By = Utils.mkvc(By,2)

    if Bz.ndim is 1:
        Bz = Utils.mkvc(Bz,2)

    return Bx, By, Bz


def ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', mu=mu_0):
    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)

    dx = XYZ[:,0]-srcLoc[0]
    dy = XYZ[:,1]-srcLoc[1]
    dz = XYZ[:,2]-srcLoc[2]

    r  = np.sqrt( dx**2. + dy**2. + dz**2.)
    k  = np.sqrt( -1j*2.*np.pi*f*mu*sig )
    kr = k*r

    front = current * length / (4. * np.pi * sig * r**3) * np.exp(-1j*k*r)
    mid   = -k**2 * r**2 + 3*1j*k*r + 3

    # Ex = front*((dx**2 / r**2)*mid + (k**2 * r**2 -1j*k*r))
    # Ey = front*(dx*dy  / r**2)*mid
    # Ez = front*(dx*dz  / r**2)*mid

    if orientation.upper() == 'X':
        Ex = front*((dx**2 / r**2)*mid + (k**2 * r**2 -1j*k*r-1.))
        Ey = front*(dx*dy  / r**2)*mid
        Ez = front*(dx*dz  / r**2)*mid
        return Ex, Ey, Ez

    elif orientation.upper() == 'Y':
        #  x--> y, y--> z, z-->x
        Ey = front*((dy**2 / r**2)*mid + (k**2 * r**2 -1j*k*r-1.))
        Ez = front*(dy*dz  / r**2)*mid
        Ex = front*(dy*dx  / r**2)*mid
        return Ex, Ey, Ez

    elif orientation.upper() == 'Z':
        # x --> z, y --> x, z --> y
        Ez = front*((dz**2 / r**2)*mid + (k**2 * r**2 -1j*k*r-1.))
        Ex = front*(dz*dx  / r**2)*mid
        Ey = front*(dz*dy  / r**2)*mid
        return Ex, Ey, Ez
        # return Ey, Ez, Ex
