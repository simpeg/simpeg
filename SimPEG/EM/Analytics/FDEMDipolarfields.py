from __future__ import division
import numpy as np
from scipy.constants import mu_0, pi, epsilon_0
from scipy.special import erf
from SimPEG import Utils

omega = lambda f: 2.*np.pi*f
# TODO:
# r = lambda dx, dy, dz: np.sqrt( dx**2. + dy**2. + dz**2.)
# k = lambda f, mu, epsilon, sig: np.sqrt( omega(f)**2. *mu*epsilon -1j*omega(f)*mu*sig )

def E_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=0., epsr=1.):

    """
        Computing Analytic Electric fields from Electrical Dipole in a Wholespace
        TODO:
            Add description of parameters
    """
    mu = mu_0*(1+kappa)
    epsilon = epsilon_0*epsr
    sig_hat = sig + 1j*omega(f)*epsilon

    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & f.shape[0] > 1:
        raise Exception("I/O type error: For multiple field locations only a single frequency can be specified.")

    dx = XYZ[:,0]-srcLoc[0]
    dy = XYZ[:,1]-srcLoc[1]
    dz = XYZ[:,2]-srcLoc[2]

    r  = np.sqrt( dx**2. + dy**2. + dz**2.)
    # k  = np.sqrt( -1j*2.*np.pi*f*mu*sig )
    k  = np.sqrt( omega(f)**2. *mu*epsilon -1j*omega(f)*mu*sig )

    front = current * length / (4.*np.pi*sig_hat* r**3) * np.exp(-1j*k*r)
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


def E_galvanic_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=1., epsr=1.):

    """
        Computing Galvanic portion of Electric fields from Electrical Dipole in a Wholespace
        TODO:
            Add description of parameters
    """
    mu = mu_0*(1+kappa)
    epsilon = epsilon_0*epsr
    sig_hat = sig + 1j*omega(f)*epsilon

    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & f.shape[0] > 1:
        raise Exception("I/O type error: For multiple field locations only a single frequency can be specified.")

    dx = XYZ[:,0]-srcLoc[0]
    dy = XYZ[:,1]-srcLoc[1]
    dz = XYZ[:,2]-srcLoc[2]

    r  = np.sqrt( dx**2. + dy**2. + dz**2.)
    # k  = np.sqrt( -1j*2.*np.pi*f*mu*sig )
    k  = np.sqrt( omega(f)**2. *mu*epsilon -1j*omega(f)*mu*sig )

    front = current * length / (4.*np.pi*sig_hat* r**3) * np.exp(-1j*k*r)
    mid   = -k**2 * r**2 + 3*1j*k*r + 3

    if orientation.upper() == 'X':
        Ex_galvanic = front*((dx**2 / r**2)*mid + (-1j*k*r-1.))
        Ey_galvanic = front*(dx*dy  / r**2)*mid
        Ez_galvanic = front*(dx*dz  / r**2)*mid
        return Ex_galvanic, Ey_galvanic, Ez_galvanic

    elif orientation.upper() == 'Y':
        #  x--> y, y--> z, z-->x
        Ey_galvanic = front*((dy**2 / r**2)*mid + (-1j*k*r-1.))
        Ez_galvanic = front*(dy*dz  / r**2)*mid
        Ex_galvanic = front*(dy*dx  / r**2)*mid
        return Ex_galvanic, Ey_galvanic, Ez_galvanic

    elif orientation.upper() == 'Z':
        # x --> z, y --> x, z --> y
        Ez_galvanic = front*((dz**2 / r**2)*mid + (-1j*k*r-1.))
        Ex_galvanic = front*(dz*dx  / r**2)*mid
        Ey_galvanic = front*(dz*dy  / r**2)*mid
        return Ex_galvanic, Ey_galvanic, Ez_galvanic


def E_inductive_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=1., epsr=1.):

    """
        Computing Inductive portion of Electric fields from Electrical Dipole in a Wholespace
        TODO:
            Add description of parameters
    """
    mu = mu_0*(1+kappa)
    epsilon = epsilon_0*epsr
    sig_hat = sig + 1j*omega(f)*epsilon

    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & f.shape[0] > 1:
        raise Exception("I/O type error: For multiple field locations only a single frequency can be specified.")

    dx = XYZ[:,0]-srcLoc[0]
    dy = XYZ[:,1]-srcLoc[1]
    dz = XYZ[:,2]-srcLoc[2]

    r  = np.sqrt( dx**2. + dy**2. + dz**2.)
    # k  = np.sqrt( -1j*2.*np.pi*f*mu*sig )
    k  = np.sqrt( omega(f)**2. *mu*epsilon -1j*omega(f)*mu*sig )

    front = current * length / (4.*np.pi*sig_hat* r**3) * np.exp(-1j*k*r)

    if orientation.upper() == 'X':
        Ex_inductive = front*(k**2 * r**2)
        Ey_inductive = np.zeros_like(Ex_inductive)
        Ez_inductive = np.zeros_like(Ex_inductive)
        return Ex_inductive, Ey_inductive, Ez_inductive

    elif orientation.upper() == 'Y':
        #  x--> y, y--> z, z-->x
        Ey_inductive = front*(k**2 * r**2)
        Ez_inductive = np.zeros_like(Ey_inductive)
        Ex_inductive = np.zeros_like(Ey_inductive)
        return Ex_inductive, Ey_inductive, Ez_inductive

    elif orientation.upper() == 'Z':
        # x --> z, y --> x, z --> y
        Ez_inductive = front*(k**2 * r**2)
        Ex_inductive = np.zeros_like(Ez_inductive)
        Ey_inductive = np.zeros_like(Ez_inductive)
        return Ex_inductive, Ey_inductive, Ez_inductive


def J_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=1., epsr=1.):

    """
        Computing Current densities from Electrical Dipole in a Wholespace
        TODO:
            Add description of parameters
    """

    Ex, Ey, Ez = E_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=current, length=length, orientation=orientation, kappa=kappa, epsr=epsr)
    Jx = sig*Ex
    Jy = sig*Ey
    Jz = sig*Ez
    return Jx, Jy, Jz


def J_galvanic_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=1., epsr=1.):

    """
        Computing Galvanic portion of Current densities from Electrical Dipole in a Wholespace
        TODO:
            Add description of parameters
    """

    Ex_galvanic, Ey_galvanic, Ez_galvanic = E_galvanic_from_ElectricDipoleWholeSpaced(XYZ, srcLoc, sig, f, current=current, length=length, orientation=orientation, kappa=kappa, epsr=epsr)
    Jx_galvanic = sig*Ex_galvanic
    Jy_galvanic = sig*Ey_galvanic
    Jz_galvanic = sig*Ez_galvanic
    return Jx_galvanic, Jy_galvanic, Jz_galvanic


def J_inductive_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=1., epsr=1.):

    """
        Computing Inductive portion of Current densities from Electrical Dipole in a Wholespace
        TODO:
            Add description of parameters
    """

    Ex_inductive, Ey_inductive, Ez_inductive = E_inductive_from_ElectricDipoleWholeSpaced(XYZ, srcLoc, sig, f, current=current, length=length, orientation=orientation, kappa=kappa, epsr=epsr)
    Jx_inductive = sig*Ex_inductive
    Jy_inductive = sig*Ey_inductive
    Jz_inductive = sig*Ez_inductive
    return Jx_inductive, Jy_inductive, Jz_inductive


def H_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=1., epsr=1.):

    """
        Computing Magnetic fields from Electrical Dipole in a Wholespace
        TODO:
            Add description of parameters
    """
    mu = mu_0*(1+kappa)
    epsilon = epsilon_0*epsr
    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & f.shape[0] > 1:
        raise Exception("I/O type error: For multiple field locations only a single frequency can be specified.")

    dx = XYZ[:,0]-srcLoc[0]
    dy = XYZ[:,1]-srcLoc[1]
    dz = XYZ[:,2]-srcLoc[2]

    r  = np.sqrt( dx**2. + dy**2. + dz**2.)
    # k  = np.sqrt( -1j*2.*np.pi*f*mu*sig )
    k  = np.sqrt( omega(f)**2. *mu*epsilon -1j*omega(f)*mu*sig )

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


def B_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=1., epsr=1.):

    """
        Computing Magnetic flux densites from Electrical Dipole in a Wholespace
        TODO:
            Add description of parameters
    """

    Hx, Hy, Hz = H_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=current, length=length, orientation=orientation, kappa=kappa, epsr=epsr)
    Bx = mu*Hx
    By = mu*Hy
    Bz = mu*Hz
    return Bx, By, Bz


def A_from_ElectricDipoleWholeSpace(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=1., epsr=1.):

    """
        Computing Electric vector potentials from Electrical Dipole in a Wholespace
        TODO:
            Add description of parameters
    """
    mu = mu_0*(1+kappa)
    epsilon = epsilon_0*epsr
    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & f.shape[0] > 1:
        raise Exception("I/O type error: For multiple field locations only a single frequency can be specified.")

    dx = XYZ[:,0]-srcLoc[0]
    dy = XYZ[:,1]-srcLoc[1]
    dz = XYZ[:,2]-srcLoc[2]

    r  = np.sqrt( dx**2. + dy**2. + dz**2.)
    k  = np.sqrt( omega(f)**2. *mu*epsilon -1j*omega(f)*mu*sig )

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





