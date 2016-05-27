from __future__ import division
import numpy as np
from scipy.constants import mu_0, pi
from scipy.special import erf
from SimPEG import Utils


def E_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', mu=mu_0):
    epsilon = 8.854187817*(10.**-12)
    omega = 2.*np.pi*f

    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & f.length > 1
        except:
            print "I/O type error: For multiple field locations only a single frequency can be specified."

    dx = XYZ[:,0]-srcLoc[0]
    dy = XYZ[:,1]-srcLoc[1]
    dz = XYZ[:,2]-srcLoc[2]

    r  = np.sqrt( dx**2. + dy**2. + dz**2.)
    # k  = np.sqrt( -1j*2.*np.pi*f*mu*sig )
    k  = np.sqrt( omega**2. *mu*epsilon -1j*omega*mu*sig )

    front = current * length / (4.*np.pi*sig* r**3) * np.exp(-1j*k*r)
    mid   = -k**2 * r**2 + 3*1j*k*r + 3

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

def J_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', mu=mu_0):
    Ex, Ey, Ez = E_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', mu=mu_0)
    Jx = sig*Ex
    Jy = sig*Ey
    Jz = sig*Ez
    return Jx, Jy, Jz        


def H_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', mu=mu_0):
    epsilon = 8.854187817*(10.**-12)
    omega = 2.*np.pi*f

    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & f.length > 1
        except:
            print "I/O type error: For multiple field locations only a single frequency can be specified."
    
    dx = XYZ[:,0]-srcLoc[0]
    dy = XYZ[:,1]-srcLoc[1]
    dz = XYZ[:,2]-srcLoc[2]

    r  = np.sqrt( dx**2. + dy**2. + dz**2.)
    # k  = np.sqrt( -1j*2.*np.pi*f*mu*sig )
    k  = np.sqrt( omega**2. *mu*epsilon -1j*omega*mu*sig )

    front = current * length / (4.*np.pi* r**2) * (-1j*k*r + 1) * np.exp(-1j*k*r)

    if orientation.upper() == 'X':
        Hy = front*(-dz  / r)
        Hz = front*(dy  / r)
        Hx = np.zeros_like(Hy)
        return Hx, Hy, Hz

    elif orientation.upper() == 'Y':
        Hx = front*(dz  / r)
        Hz = front*(-dx  / r)
        Hy = np.zeros_like(Hx)
        return Hx, Hy, Hz

    elif orientation.upper() == 'Z':
        Hx = front*(-dy / r)
        Hy = front*(dx  / r)
        Hz = np.zeros_like(Hx)
        return Hx, Hy, Hz


def B_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', mu=mu_0):
    Hx, Hy, Hz = H_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', mu=mu_0)
    Bx = mu*Hx
    By = mu*Hy
    Bz = mu*Hz
    return Bx, By, Bz
      

def A_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', mu=mu_0):
    epsilon = 8.854187817*(10.**-12)
    omega = 2.*np.pi*f

    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & f.length > 1
        except:
            print "I/O type error: For multiple field locations only a single frequency can be specified."

    dx = XYZ[:,0]-srcLoc[0]
    dy = XYZ[:,1]-srcLoc[1]
    dz = XYZ[:,2]-srcLoc[2]

    r  = np.sqrt( dx**2. + dy**2. + dz**2.)
    k  = np.sqrt( omega**2. *mu*epsilon -1j*omega*mu*sig )

    front = current * length / (4.*np.pi*r)

    if orientation.upper() == 'X':
        Ax = front*np.exp(-1j*k*r)
        Ay = np.zeros_like(Ax)
        Az = np.zeros_like(Ax)
        return Ax, Ay, Az

    elif orientation.upper() == 'Y':
        Ay = front*np.exp(-1j*k*r)
        Ax = np.zeros_like(Ay)
        Az = np.zeros_like(Ay)
        return Ax, Ay, Az

    elif orientation.upper() == 'Z':
        Az = front*np.exp(-1j*k*r)
        Ax = np.zeros_like(Ay)
        Ay = np.zeros_like(Ay)
        return Ax, Ay, Az





