from __future__ import division
import numpy as np
from scipy.constants import mu_0, pi, epsilon_0
from scipy.special import erf
from SimPEG import Utils

omega = lambda f: 2.*np.pi*f
# TODO:
# r = lambda dx, dy, dz: np.sqrt( dx**2. + dy**2. + dz**2.)
# k = lambda f, mu, epsilon, sig: np.sqrt( omega(f)**2. *mu*epsilon -1j*omega(f)*mu*sig )

def ElectricDipoleWholeSpace_E(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=0., epsr=1., t=0.):

    """
        Computing the analytic electric fields (E) from an electrical dipole in a wholespace
        - You have the option of computing E for multiple frequencies at a single reciever location
          or a single frequency at multiple locations

        :param numpy.array XYZ: reciever locations at which to evaluate E
        :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
        :param float sig: value specifying the conductivity (S/m) of the wholespace
        :param numpy.array f: array of Tx frequencies (Hz)
        :param float current: size of the injected current (A), default is 1.0 A
        :param float length: length of the dipole (m), default is 1.0 m
        :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
        :param float kappa: magnetic susceptiblity value (unitless), default is 0.
        :param float epsr: relative permitivitty value (unitless),  default is 1.0
        :param float t: time variable, only used for Seogi's plotting application...
        :rtype: numpy.array
        :return: Ex, Ey, Ez: arrays containing all 3 components of E evaluated at the specified locations and frequencies.
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


def ElectricDipoleWholeSpace_E_galvanic(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=1., epsr=1., t=0.):

    """
        Computing the galvanic portion of the analytic electric fields (E) from an electrical dipole in a wholespace
        - You have the option of computing E for multiple frequencies at a single reciever location
          or a single frequency at multiple locations

        :param numpy.array XYZ: reciever locations at which to evaluate E_galvanic
        :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
        :param float sig: value specifying the conductivity (S/m) of the wholespace
        :param numpy.array f: array of Tx frequencies (Hz)
        :param float current: size of the injected current (A), default is 1.0 A
        :param float length: length of the dipole (m), default is 1.0 m
        :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
        :param float kappa: magnetic susceptiblity value (unitless), default is 0.
        :param float epsr: relative permitivitty value (unitless),  default is 1.0
        :param float t: time variable, only used for Seogi's plotting application...
        :rtype: numpy.array
        :return: Ex, Ey, Ez: arrays containing the galvanic portion of all 3 components of E evaluated at the specified locations and frequencies.
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


def ElectricDipoleWholeSpace_E_inductive(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=1., epsr=1., t=0.):

    """
        Computing the inductive portion of the analytic electric fields (E) from an electrical dipole in a wholespace
        - You have the option of computing E for multiple frequencies at a single reciever location
          or a single frequency at multiple locations

        :param numpy.array XYZ: reciever locations at which to evaluate E_inductive
        :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
        :param float sig: value specifying the conductivity (S/m) of the wholespace
        :param numpy.array f: array of Tx frequencies (Hz)
        :param float current: size of the injected current (A), default is 1.0 A
        :param float length: length of the dipole (m), default is 1.0 m
        :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
        :param float kappa: magnetic susceptiblity value (unitless), default is 0.
        :param float epsr: relative permitivitty value (unitless),  default is 1.0
        :param float t: time variable, only used for Seogi's plotting application...
        :rtype: numpy.array
        :return: Ex, Ey, Ez: arrays containing the inductive portion of all 3 components of E evaluated at the specified locations and frequencies.
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


def ElectricDipoleWholeSpace_J(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=1., epsr=1., t=0.):

    """
        Computing the analytic current density (J) from an electrical dipole in a wholespace
        - You have the option of computing J for multiple frequencies at a single reciever location
          or a single frequency at multiple locations

        :param numpy.array XYZ: reciever locations at which to evaluate J
        :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
        :param float sig: value specifying the conductivity (S/m) of the wholespace
        :param numpy.array f: array of Tx frequencies (Hz)
        :param float current: size of the injected current (A), default is 1.0 A
        :param float length: length of the dipole (m), default is 1.0 m
        :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
        :param float kappa: magnetic susceptiblity value (unitless), default is 0.
        :param float epsr: relative permitivitty value (unitless),  default is 1.0
        :param float t: time variable, only used for Seogi's plotting application...
        :rtype: numpy.array
        :return: Jx, Jy, Jz: arrays containing all 3 components of J evaluated at the specified locations and frequencies.
    """

    Ex, Ey, Ez = ElectricDipoleWholeSpace_E(XYZ, srcLoc, sig, f, current=current, length=length, orientation=orientation, kappa=kappa, epsr=epsr)
    Jx = sig*Ex
    Jy = sig*Ey
    Jz = sig*Ez
    return Jx, Jy, Jz


def ElectricDipoleWholeSpace_J_galvanic(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=1., epsr=1., t=0.):

    """
        Computing the galvanic portion of the analytic current density (J) from an electrical dipole in a wholespace
        - You have the option of computing J for multiple frequencies at a single reciever location
          or a single frequency at multiple locations

        :param numpy.array XYZ: reciever locations at which to evaluate J_galvanic
        :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
        :param float sig: value specifying the conductivity (S/m) of the wholespace
        :param numpy.array f: array of Tx frequencies (Hz)
        :param float current: size of the injected current (A), default is 1.0 A
        :param float length: length of the dipole (m), default is 1.0 m
        :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
        :param float kappa: magnetic susceptiblity value (unitless), default is 0.
        :param float epsr: relative permitivitty value (unitless),  default is 1.0
        :param float t: time variable, only used for Seogi's plotting application...
        :rtype: numpy.array
        :return: Jx, Jy, Jz: arrays containing the galvanic portion of all 3 components of J evaluated at the specified locations and frequencies.
    """

    Ex_galvanic, Ey_galvanic, Ez_galvanic = ElectricDipoleWholeSpace_E_galvanic(XYZ, srcLoc, sig, f, current=current, length=length, orientation=orientation, kappa=kappa, epsr=epsr)
    Jx_galvanic = sig*Ex_galvanic
    Jy_galvanic = sig*Ey_galvanic
    Jz_galvanic = sig*Ez_galvanic
    return Jx_galvanic, Jy_galvanic, Jz_galvanic


def ElectricDipoleWholeSpace_J_inductive(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=1., epsr=1., t=0.):

    """
        Computing the inductive portion of the analytic current density (J) from an electrical dipole in a wholespace
        - You have the option of computing J for multiple frequencies at a single reciever location
          or a single frequency at multiple locations

        :param numpy.array XYZ: reciever locations at which to evaluate J_inductive
        :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
        :param float sig: value specifying the conductivity (S/m) of the wholespace
        :param numpy.array f: array of Tx frequencies (Hz)
        :param float current: size of the injected current (A), default is 1.0 A
        :param float length: length of the dipole (m), default is 1.0 m
        :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
        :param float kappa: magnetic susceptiblity value (unitless), default is 0.
        :param float epsr: relative permitivitty value (unitless),  default is 1.0
        :param float t: time variable, only used for Seogi's plotting application...
        :rtype: numpy.array
        :return: Jx, Jy, Jz: arrays containing the galvanic portion of all 3 components of J evaluated at the specified locations and frequencies.
    """

    Ex_inductive, Ey_inductive, Ez_inductive = ElectricDipoleWholeSpace_E_inductive(XYZ, srcLoc, sig, f, current=current, length=length, orientation=orientation, kappa=kappa, epsr=epsr)
    Jx_inductive = sig*Ex_inductive
    Jy_inductive = sig*Ey_inductive
    Jz_inductive = sig*Ez_inductive
    return Jx_inductive, Jy_inductive, Jz_inductive


def ElectricDipoleWholeSpace_H(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=1., epsr=1., t=0.):

    """
        Computing the analytic magnetic fields (H) from an electrical dipole in a wholespace
        - You have the option of computing H for multiple frequencies at a single reciever location
          or a single frequency at multiple locations

        :param numpy.array XYZ: reciever locations at which to evaluate H
        :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
        :param float sig: value specifying the conductivity (S/m) of the wholespace
        :param numpy.array f: array of Tx frequencies (Hz)
        :param float current: size of the injected current (A), default is 1.0 A
        :param float length: length of the dipole (m), default is 1.0 m
        :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
        :param float kappa: magnetic susceptiblity value (unitless), default is 0.
        :param float epsr: relative permitivitty value (unitless),  default is 1.0
        :param float t: time variable, only used for Seogi's plotting application...
        :rtype: numpy.array
        :return: Hx, Hy, Hz: arrays containing all 3 components of H evaluated at the specified locations and frequencies.
    """

    mu = mu_0*(1+kappa)
    epsilon = epsilon_0*epsr
    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & f.shape[0] > 1:
        raise Exception("I/O type error: For multiple field locations only a single frequency can be specified.")

    dx = XYZ[:, 0]-srcLoc[0]
    dy = XYZ[:, 1]-srcLoc[1]
    dz = XYZ[:, 2]-srcLoc[2]

    r = np.sqrt(dx**2. + dy**2. + dz**2.)
    # k  = np.sqrt( -1j*2.*np.pi*f*mu*sig )
    k = np.sqrt(omega(f)**2. * mu*epsilon - 1j*omega(f)*mu*sig)

    front = current * length / (4.*np.pi*(r)**2) * (1j*k*r + 1) * np.exp(-1j*k*r)

    if orientation.upper() == 'X':
        Hy = front*(-dz / r)
        Hz = front*(dy / r)
        Hx = np.zeros_like(Hy)
        return Hx, Hy, Hz

    elif orientation.upper() == 'Y':
        Hx = front*(dz / r)
        Hz = front*(-dx / r)
        Hy = np.zeros_like(Hx)
        return Hx, Hy, Hz

    elif orientation.upper() == 'Z':
        Hx = front*(-dy / r)
        Hy = front*(dx / r)
        Hz = np.zeros_like(Hx)
        return Hx, Hy, Hz


def ElectricDipoleWholeSpace_B(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=1., epsr=1., t=0.):

    """
        Computing the analytic magnetic flux density (B) from an electrical dipole in a wholespace
        - You have the option of computing B for multiple frequencies at a single reciever location
          or a single frequency at multiple locations

        :param numpy.array XYZ: reciever locations at which to evaluate B
        :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
        :param float sig: value specifying the conductivity (S/m) of the wholespace
        :param numpy.array f: array of Tx frequencies (Hz)
        :param float current: size of the injected current (A), default is 1.0 A
        :param float length: length of the dipole (m), default is 1.0 m
        :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
        :param float kappa: magnetic susceptiblity value (unitless), default is 0.
        :param float epsr: relative permitivitty value (unitless),  default is 1.0
        :param float t: time variable, only used for Seogi's plotting application...
        :rtype: numpy.array
        :return: Bx, By, Bz: arrays containing all 3 components of B evaluated at the specified locations and frequencies.
    """

    mu = mu_0*(1+kappa)

    Hx, Hy, Hz = ElectricDipoleWholeSpace_H(XYZ, srcLoc, sig, f, current=current, length=length, orientation=orientation, kappa=kappa, epsr=epsr)
    Bx = mu*Hx
    By = mu*Hy
    Bz = mu*Hz
    return Bx, By, Bz


def ElectricDipoleWholeSpace_A(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=1., epsr=1., t=0.):

    """
        Computing the analytic electric vector potential (A) from an electrical dipole in a wholespace
        - You have the option of computing A for multiple frequencies at a single reciever location
          or a single frequency at multiple locations

        :param numpy.array XYZ: reciever locations at which to evaluate A
        :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
        :param float sig: value specifying the conductivity (S/m) of the wholespace
        :param numpy.array f: array of Tx frequencies (Hz)
        :param float current: size of the injected current (A), default is 1.0 A
        :param float length: length of the dipole (m), default is 1.0 m
        :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
        :param float kappa: magnetic susceptiblity value (unitless), default is 0.
        :param float epsr: relative permitivitty value (unitless),  default is 1.0
        :param float t: time variable, only used for Seogi's plotting application...
        :rtype: numpy.array
        :return: Ax, Ay, Az: arrays containing all 3 components of A evaluated at the specified locations and frequencies.
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


def MagneticDipoleWholeSpace_E(XYZ, srcLoc, sig, f, current=1., loopArea=1., orientation='X', kappa=0., epsr=1., t=0.):

    """
        Computing the analytic electric fields (E) from a magnetic dipole in a wholespace
        - You have the option of computing E for multiple frequencies at a single reciever location
          or a single frequency at multiple locations

        :param numpy.array XYZ: reciever locations at which to evaluate E
        :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
        :param float sig: value specifying the conductivity (S/m) of the wholespace
        :param numpy.array f: array of Tx frequencies (Hz)
        :param float current: size of the injected current (A), default is 1.0 A
        :param float length: length of the dipole (m), default is 1.0 m
        :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
        :param float kappa: magnetic susceptiblity value (unitless), default is 0.
        :param float epsr: relative permitivitty value (unitless),  default is 1.0
        :param float t: time variable, only used for Seogi's plotting application...
        :rtype: numpy.array
        :return: Ex, Ey, Ez: arrays containing all 3 components of E evaluated at the specified locations and frequencies.
    """

    mu = mu_0 * (1+kappa)
    epsilon = epsilon_0 * epsr
    m = current * loopArea

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

    front = ((1j * omega(f) * mu * m) / (4.* np.pi * r**2)) * (1j * k * r + 1) * np.exp(-1j*k*r)

    if orientation.upper() == 'X':
        Ey = front * (dz / r)
        Ez = front * (-dy / r)
        Ex = np.zeros_like(Ey)
        return Ex, Ey, Ez

    elif orientation.upper() == 'Y':
        Ex = front * (-dz / r)
        Ez = front * (dx / r)
        Ey = np.zeros_like(Ex)
        return Ex, Ey, Ez

    elif orientation.upper() == 'Z':
        Ex = front * (dy / r)
        Ey = front * (-dx / r)
        Ez = np.zeros_like(Ex)
        return Ex, Ey, Ez


def MagneticDipoleWholeSpace_J(XYZ, srcLoc, sig, f, current=1., loopArea=1., orientation='X', kappa=1., epsr=1., t=0.):

    """
        Computing the analytic current density (J) from a magnetic dipole in a wholespace
        - You have the option of computing J for multiple frequencies at a single reciever location
          or a single frequency at multiple locations

        :param numpy.array XYZ: reciever locations at which to evaluate J
        :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
        :param float sig: value specifying the conductivity (S/m) of the wholespace
        :param numpy.array f: array of Tx frequencies (Hz)
        :param float current: size of the injected current (A), default is 1.0 A
        :param float length: length of the dipole (m), default is 1.0 m
        :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
        :param float kappa: magnetic susceptiblity value (unitless), default is 0.
        :param float epsr: relative permitivitty value (unitless),  default is 1.0
        :param float t: time variable, only used for Seogi's plotting application...
        :rtype: numpy.array
        :return: Jx, Jy, Jz: arrays containing all 3 components of J evaluated at the specified locations and frequencies.
    """

    Ex, Ey, Ez = MagneticDipoleWholeSpace_E(XYZ, srcLoc, sig, f, current=current, loopArea=loopArea, orientation=orientation, kappa=kappa, epsr=epsr)
    Jx = sig * Ex
    Jy = sig * Ey
    Jz = sig * Ez
    return Jx, Jy, Jz


def MagneticDipoleWholeSpace_H(XYZ, srcLoc, sig, f, current=1., loopArea=1., orientation='X', kappa=1., epsr=1., t=0.):

    """
        Computing the analytic magnetic fields (H) from a magnetic dipole in a wholespace
        - You have the option of computing H for multiple frequencies at a single reciever location
          or a single frequency at multiple locations

        :param numpy.array XYZ: reciever locations at which to evaluate H
        :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
        :param float sig: value specifying the conductivity (S/m) of the wholespace
        :param numpy.array f: array of Tx frequencies (Hz)
        :param float current: size of the injected current (A), default is 1.0 A
        :param float length: length of the dipole (m), default is 1.0 m
        :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
        :param float kappa: magnetic susceptiblity value (unitless), default is 0.
        :param float epsr: relative permitivitty value (unitless),  default is 1.0
        :param float t: time variable, only used for Seogi's plotting application...
        :rtype: numpy.array
        :return: Hx, Hy, Hz: arrays containing all 3 components of H evaluated at the specified locations and frequencies.
    """

    mu = mu_0 * (1+kappa)
    epsilon = epsilon_0 * epsr
    m = current * loopArea

    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & f.shape[0] > 1:
        raise Exception("I/O type error: For multiple field locations only a single frequency can be specified.")

    dx = XYZ[:, 0]-srcLoc[0]
    dy = XYZ[:, 1]-srcLoc[1]
    dz = XYZ[:, 2]-srcLoc[2]

    r = np.sqrt(dx**2. + dy**2. + dz**2.)
    # k  = np.sqrt( -1j*2.*np.pi*f*mu*sig )
    k = np.sqrt(omega(f)**2. * mu*epsilon - 1j*omega(f)*mu*sig)

    front = m / (4.*np.pi*(r)**3) * np.exp(-1j*k*r)
    mid   = -k**2 * r**2 + 3*1j*k*r + 3

    if orientation.upper() == 'X':
        Hx = front*((dx**2 / r**2)*mid + (k**2 * r**2 - 1j*k*r - 1.))
        Hy = front*(dx*dy  / r**2)*mid
        Hz = front*(dx*dz  / r**2)*mid
        return Hx, Hy, Hz

    elif orientation.upper() == 'Y':
        #  x--> y, y--> z, z-->x
        Hy = front * ((dy**2 / r**2)*mid + (k**2 * r**2 - 1j*k*r - 1.))
        Hz = front * (dy*dz  / r**2)*mid
        Hx = front * (dy*dx  / r**2)*mid
        return Hx, Hy, Hz

    elif orientation.upper() == 'Z':
        # x --> z, y --> x, z --> y
        Hz = front*((dz**2 / r**2)*mid + (k**2 * r**2 - 1j*k*r - 1.))
        Hx = front*(dz*dx  / r**2)*mid
        Hy = front*(dz*dy  / r**2)*mid
        return Hx, Hy, Hz


def MagneticDipoleWholeSpace_B(XYZ, srcLoc, sig, f, current=1., loopArea=1., orientation='X', kappa=1., epsr=1., t=0.):

    """
        Computing the analytic magnetic flux density (B) from a magnetic dipole in a wholespace
        - You have the option of computing B for multiple frequencies at a single reciever location
          or a single frequency at multiple locations

        :param numpy.array XYZ: reciever locations at which to evaluate B
        :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
        :param float sig: value specifying the conductivity (S/m) of the wholespace
        :param numpy.array f: array of Tx frequencies (Hz)
        :param float current: size of the injected current (A), default is 1.0 A
        :param float length: length of the dipole (m), default is 1.0 m
        :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
        :param float kappa: magnetic susceptiblity value (unitless), default is 0.
        :param float epsr: relative permitivitty value (unitless),  default is 1.0
        :param float t: time variable, only used for Seogi's plotting application...
        :rtype: numpy.array
        :return: Bx, By, Bz: arrays containing all 3 components of B evaluated at the specified locations and frequencies.
    """

    mu = mu_0 * (1+kappa)

    Hx, Hy, Hz = MagneticDipoleWholeSpace_H(XYZ, srcLoc, sig, f, current=current, loopArea=loopArea, orientation=orientation, kappa=kappa, epsr=epsr)
    Bx = mu * Hx
    By = mu * Hy
    Bz = mu * Hz
    return Bx, By, Bz


def MagneticDipoleWholeSpace_F(XYZ, srcLoc, sig, f, current=1., loopArea=1., orientation='X', kappa=1., epsr=1., t=0.):

    """
        Computing the analytic magnetic vector potential (F) from a magnetic dipole in a wholespace
        - You have the option of computing F for multiple frequencies at a single reciever location
          or a single frequency at multiple locations

        :param numpy.array XYZ: reciever locations at which to evaluate F
        :param numpy.array srcLoc: [x,y,z] triplet defining the location of the electric dipole source
        :param float sig: value specifying the conductivity (S/m) of the wholespace
        :param numpy.array f: array of Tx frequencies (Hz)
        :param float current: size of the injected current (A), default is 1.0 A
        :param float length: length of the dipole (m), default is 1.0 m
        :param str orientation: orientation of dipole: 'X', 'Y', or 'Z'
        :param float kappa: magnetic susceptiblity value (unitless), default is 0.
        :param float epsr: relative permitivitty value (unitless),  default is 1.0
        :param float t: time variable, only used for Seogi's plotting application...
        :rtype: numpy.array
        :return: Fx, Fy, Fz: arrays containing all 3 components of F evaluated at the specified locations and frequencies.
    """

    mu = mu_0 * (1+kappa)
    epsilon = epsilon_0*epsr
    m = current * loopArea

    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & f.shape[0] > 1:
        raise Exception("I/O type error: For multiple field locations only a single frequency can be specified.")

    dx = XYZ[:,0]-srcLoc[0]
    dy = XYZ[:,1]-srcLoc[1]
    dz = XYZ[:,2]-srcLoc[2]

    r  = np.sqrt( dx**2. + dy**2. + dz**2.)
    k  = np.sqrt( omega(f)**2. *mu*epsilon -1j*omega(f)*mu*sig )

    front = (1j * omega(f) * mu * m) / (4.* np.pi * r)

    if orientation.upper() == 'X':
        Fx = front*np.exp(-1j*k*r)
        Fy = np.zeros_like(Fx)
        Fz = np.zeros_like(Fx)
        return Fx, Fy, Fz

    elif orientation.upper() == 'Y':
        Fy = front*np.exp(-1j*k*r)
        Fx = np.zeros_like(Fy)
        Fz = np.zeros_like(Fy)
        return Fx, Fy, Fz

    elif orientation.upper() == 'Z':
        Fz = front*np.exp(-1j*k*r)
        Fx = np.zeros_like(Fy)
        Fy = np.zeros_like(Fy)
        return Fx, Fy, Fz



