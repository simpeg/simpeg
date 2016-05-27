from __future__ import division
import numpy as np
from scipy.constants import mu_0, pi
from scipy.special import erf
from SimPEG import Utils


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

def A_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', mu=mu_0):
def J_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', mu=mu_0):
def H_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', mu=mu_0):
def B_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', mu=mu_0):
    mu*H_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', mu=mu_0)
