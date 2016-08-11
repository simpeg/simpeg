from __future__ import division
import numpy as np
from scipy.constants import mu_0, pi, epsilon_0
import numpy as np
from SimPEG import Utils


omega = lambda f: 2.*np.pi*f

def E_field_from_SheetCurruent(XYZ, srcLoc, sig, f, E0=1., orientation='X', kappa=0., epsr=1., t=0.):
    """
        Computing Analytic Electric fields from Plane wave in a Wholespace
        TODO:
            Add description of parameters
    """

    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & f.shape[0] > 1:
        raise Exception("I/O type error: For multiple field locations only a single frequency can be specified.")

    mu = mu_0*(1+kappa)
    epsilon = epsilon_0*epsr
    sig_hat = sig + 1j*omega(f)*epsilon
    k  = np.sqrt( omega(f)**2. *mu*epsilon -1j*omega(f)*mu*sig )
    # print t
    if orientation == "X":
        z = XYZ[:,2]
        Ex = E0*np.exp(1j*(k*(z-srcLoc)+omega(f)*t))
        Ey = np.zeros_like(z)
        Ez = np.zeros_like(z)
        return Ex, Ey, Ez
    else:
        raise NotImplementedError()

def J_field_from_SheetCurruent(XYZ, srcLoc, sig, f, E0=1., orientation='X', kappa=0., epsr=1., t=0.):
    """
        Plane wave propagating downward (negative z (depth))
    """

    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & f.shape[0] > 1:
        raise Exception("I/O type error: For multiple field locations only a single frequency can be specified.")

    mu = mu_0*(1+kappa)
    epsilon = epsilon_0*epsr
    sig_hat = sig + 1j*omega(f)*epsilon
    k  = np.sqrt( omega(f)**2. *mu*epsilon -1j*omega(f)*mu*sig )

    if orientation == "X":
        z = XYZ[:,2]
        Jx = sig*E0*np.exp(1j*(k*(z-srcLoc)+omega(f)*t))
        Jy = np.zeros_like(z)
        Jz = np.zeros_like(z)
        return Jx, Jy, Jz
    else:
        raise NotImplementedError()

def H_field_from_SheetCurruent(XYZ, srcLoc, sig, f, E0=1., orientation='X', kappa=0., epsr=1., t=0.):
    """
        Plane wave propagating downward (negative z (depth))
    """

    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & f.shape[0] > 1:
        raise Exception("I/O type error: For multiple field locations only a single frequency can be specified.")

    mu = mu_0*(1+kappa)
    epsilon = epsilon_0*epsr
    sig_hat = sig + 1j*omega(f)*epsilon
    k  = np.sqrt( omega(f)**2. *mu*epsilon -1j*omega(f)*mu*sig )
    Z = omega(f)*mu/k
    if orientation == "X":
        z = XYZ[:,2]
        Hx = np.zeros_like(z)
        Hy = E0/Z*np.exp(1j*(k*(z-srcLoc)+omega(f)*t))
        Hz = np.zeros_like(z)
        return Hx, Hy, Hz
    else:
        raise NotImplementedError()

def B_field_from_SheetCurruent(XYZ, srcLoc, sig, f, E0=1., orientation='X', kappa=0., epsr=1., t=0.):
    """
        Plane wave propagating downward (negative z (depth))
    """

    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & f.shape[0] > 1:
        raise Exception("I/O type error: For multiple field locations only a single frequency can be specified.")

    mu = mu_0*(1+kappa)
    epsilon = epsilon_0*epsr
    sig_hat = sig + 1j*omega(f)*epsilon
    k  = np.sqrt( omega(f)**2. *mu*epsilon -1j*omega(f)*mu*sig )
    Z = omega(f)*mu/k
    if orientation == "X":
        z = XYZ[:,2]
        Bx = mu*np.zeros_like(z)
        By = mu*E0/Z*np.exp(1j*(k*(z-srcLoc)+omega(f)*t))
        Bz = mu*np.zeros_like(z)
        return Bx, By, Bz
    else:
        raise NotImplementedError()

if __name__ == '__main__':
    pass
