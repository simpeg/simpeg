import numpy as np
from SimPEG import utils
from scipy.constants import mu_0, pi
import scipy.special as spec
import matplotlib.pyplot as plt
from .supporting_functions.digital_filter import transFiltImpulse, transFilt, setFrequency


#############################################################
#       PHYSICAL PROPERTIES
#############################################################

def ColeCole(f, sig_inf=1e-2, eta=0.1, tau=0.1, c=1):
    """
        Computing Cole-Cole model in frequency domain

        .. math ::
            \\sigma (\\omega ) = \\sigma_{\\infty} \\Bigg [
            1 - \\eta \\Bigg ( \\frac{1}{1 + (1-\\eta ) (1 + i\\omega \\tau)^c} \\Bigg )
            \\Bigg ]


    """

    if np.isscalar(sig_inf):
        w = 2*np.pi*f
        sigma = sig_inf - sig_inf*eta/(1+(1-eta)*(1j*w*tau)**c)
    else:
        sigma = np.zeros((f.size,sig_inf.size), dtype=complex)
        for i in range(f.size):
            w = 2*np.pi*f[i]
            sigma[i,:] = utils.mkvc(sig_inf - sig_inf*eta/(1+(1-eta)*(1j*w*tau)**c))
    return sigma


def LogUniform(f, chi_inf=0.05, del_chi=0.05, tau1=1e-5, tau2=1e-2):
    """
        Computing relaxation model in the frequency domain for a log-uniform
        distribution of time-relaxation constants.

        .. math::
            \\chi (\\omega ) = \\chi_{\\infty} + \\Delta \\chi \\Bigg [
            1 - \\Bigg ( \\frac{1}{ln (\\tau_2 / \\tau_1 )} \\Bigg )
            ln \\Bigg ( \\frac{1 + i\\omega \\tau_2}{1 + i\\omega tau_1} ) \\Bigg )
            \\Bigg ]


    """

    w = 2*np.pi*f
    return chi_inf + del_chi*(1 - np.log((1 + 1j*w*tau2)/(1 + 1j*w*tau1))/np.log(tau2/tau1))



def diffusion_distance(t, sigma):
    """
        Compute diffusion distance

        .. math::

            d = \sqrt{\\frac{2t}{\mu_0\sigma}}

        assume \\\\(\\\\ \mu = \mu_0\\\\) is chargeability
    """

    return np.sqrt(2*t/mu_0/sigma)


def skin_depth(f, sigma):
    """
        Compute skin depth

        .. math::

            \delta = \sqrt{\\frac{2}{\omega\mu_0\sigma}}

        where \\\\(\\\\ \omega = 2\pi f \\\\) is chargeability
    """

    return np.sqrt(2/mu_0/sigma/f/(2*pi))


#############################################################
#       VMD SOURCE SOLUTIONS
#############################################################

def Hz_vertical_magnetic_dipole(f, r, sig, flag="secondary"):

    """

        Hz component of analytic solution for half-space (VMD source)
        Src and Rx are on the surface

        .. math::

            H_z  = \\frac{m}{2\pi k^2 r^5} \
                    \left( 9 -(9+\imath\ kr - 4 k^2r^2 - \imath k^3r^3)e^{-\imath kr}\\right)

        * r: Src-Rx offset
        * m: magnetic dipole moment
        * k: propagation constant

        .. math::

            k = \omega^2\epsilon\mu - \imath\omega\mu\sigma


    """

    mu0 = 4*np.pi*1e-7
    w = 2*np.pi*f
    k = np.sqrt(-1j*w*mu0*sig)
    Hz = 1./(2*np.pi*k**2*r**5)*(9-(9+9*1j*k*r-4*k**2*r**2-1j*k**3*r**3)*np.exp(-1j*k*r))
    
    if flag == 'secondary':
        Hzp = -1/(4*np.pi*r**3)
        Hz = Hz-Hzp
    return Hz


def Hr_vertical_magnetic_dipole(f, r, sig):

    """

        Hz component of analytic solution for half-space (VMD source)
        Src and Rx are on the surface

        .. math::

            H_z  = \\frac{-mk^2}{4\pi \\rho} \
            \\Bigg [ I_1 \\Bigg ( \\frac{\i k \\rho }{2} \\Bigg ) K_1 \\Bigg ( \\frac{\i k \\rho }{2} \\Bigg )
            - I_2 \\Bigg ( \\frac{\i k \\rho }{2} \\Bigg ) K_2 \\Bigg ( \\frac{\i k \\rho }{2} \\Bigg ) \\Bigg ]

        * r: Src-Rx offset
        * m: magnetic dipole moment
        * k: propagation constant
        * :math:`I_n`: modified Bessel function of the 1st kind of order *n*
        * :math:`K_n`: modified Bessel function of the 2nd kind of order *n*

        .. math::

            k = \omega^2\epsilon\mu - \imath\omega\mu\sigma \\


    """

    mu0 = 4*np.pi*1e-7
    w = 2*np.pi*f
    k = np.sqrt(-1j*w*mu0*sig)
    alpha = 1j*k*r/2.

    IK1 = spec.iv(1, alpha)*spec.kv(1, alpha)
    IK2 = spec.iv(2, alpha)*spec.kv(2, alpha)

    Hr = (-k**2/(4*np.pi*r))*(IK1 - IK2)
    
    return Hr




def Hz_horizontal_magnetic_dipole(f, r, x, sig):

    """

        Hz component of analytic solution for half-space (HMD source)
        Src and Rx are on the surface

        .. math::

            H_z  = \\frac{mxk^2}{4\pi \\rho^2} \
            \\Bigg [ I_1 \\Bigg ( \\frac{\i k \\rho }{2} \\Bigg ) K_1 \\Bigg ( \\frac{\i k \\rho }{2} \\Bigg )
            - I_2 \\Bigg ( \\frac{\i k \\rho }{2} \\Bigg ) K_2 \\Bigg ( \\frac{\i k \\rho }{2} \\Bigg ) \\Bigg ]

        * x: Src-Rx x offset
        * r: Src-Rx offset
        * m: magnetic dipole moment
        * k: propagation constant
        * :math:`I_n`: modified Bessel function of the 1st kind of order *n*
        * :math:`K_n`: modified Bessel function of the 2nd kind of order *n*

        .. math::

            k = \omega^2\epsilon\mu - \imath\omega\mu\sigma \\


    """

    mu0 = 4*np.pi*1e-7
    w = 2*np.pi*f
    k = np.sqrt(-1j*w*mu0*sig)
    alpha = 1j*k*r/2.

    IK1 = spec.iv(1, alpha)*spec.kv(1, alpha)
    IK2 = spec.iv(2, alpha)*spec.kv(2, alpha)

    Hr = (x*k**2/(4*np.pi*r**2))*(IK1 - IK2)
    return Hr




def Bz_vertical_magnetic_dipole(r, t, sigma):

    theta = np.sqrt((sigma*mu_0)/(4*t))
    tr = theta*r
    etr = spec.erf(tr)
    t1 = (9/(2*tr**2) - 1)*etr
    t2 = (1/np.sqrt(pi))*(9/tr + 4*tr)*np.exp(-tr**2)
    hz = (t1 - t2)/(4*pi*r**3)
    return mu_0*hz



#############################################################
#       HORIZONTAL LOOP SOURCE SOLUTIONS
#############################################################


def Hz_horizontal_circular_loop(f, I, a, sig, flag="secondary"):

    """

        Hz component of analytic solution for half-space (Circular-loop source)
        Src and Rx are on the surface and receiver is located at the center of the loop.

        .. math::

            H_z  = -\\frac{I}{k^2a^3} \
                    \left( 3 -(3+\imath\ ka - k^2a^2 )e^{-\imath ka}\\right)

        * a: Src-loop radius
        * I: Current intensity

    """

    mu_0 = 4*np.pi*1e-7
    w = 2*np.pi*f
    k = np.sqrt(-1j*w*mu_0*sig)
    Hz = -I/(k**2*a**3)*(3-(3+3*1j*k*a-k**2*a**2)*np.exp(-1j*k*a))
    
    if flag == 'secondary':
        Hzp = I/2./a
        Hz = Hz-Hzp
    
    return Hz





def dHzdsiganalCirc(sig, f, I, a, flag):

    """
        Compute sensitivity for HzanalCirc by using perturbation

        .. math::

            \\frac{\partial H_z}{\partial \sigma}
            = \\frac{H_z(\sigma+\\triangle\sigma)- H_z(\sigma-\\triangle\sigma)}
                {2\\triangle\sigma}
    """
    mu_0 = 4*np.pi*1e-7
    w = 2*np.pi*f
    k = np.sqrt(-1j*w*mu_0*sig)
    perc = 0.001
    Hzfun = lambda m: HzanalCirc(m, f, I, a, flag)
    dHzdsig = (Hzfun(sig+perc*sig)-Hzfun(sig-perc*sig))/(2*perc*sig)
    return dHzdsig









def Bz_horizontal_circular_loop(a, t, sigma):
    """
        Hz component of analytic solution for half-space (Circular-loop source)
        Src and Rx are on the surface and receiver is located at the center of the loop.
        Src waveform here is step-off.

        .. math::

            h_z  = \\frac{I}{2a} \
                    \left( \\frac{3}{\sqrt{\pi}\\theta a}e^{-\\theta^2a^2}
                    +(1-\\frac{3}{2\\theta^2a^2})erf(\\theta a)\\right)

        .. math::

            \\theta = \sqrt{\\frac{\sigma\mu}{4t}}
    """

    theta = np.sqrt((sigma*mu_0)/(4*t))
    ta = theta*a
    eta = spec.erf(ta)
    t1 = (3/(np.sqrt(pi)*ta))*np.exp(-ta**2)
    t2 = (1 - (3/(2*ta**2)))*eta
    hz = (t1 + t2)/(2*a)
    return mu_0*hz


def dBzdt_horizontal_circular_loop(a, t, sigma):
    """
        Hz component of analytic solution for half-space (Circular-loop source)
        Src and Rx are on the surface and receiver is located at the center of the loop.
        Src waveform here is step-off.

        .. math::

            \\frac{\partial h_z}{\partial t}  = -\\frac{I}{\mu_0\sigma a^3} \
                    \left( 3erf(\\theta a) - \\frac{2}{\sqrt{\pi}}\\theta a (3+2\\theta^2 a^2) e^{-\\theta^2a^2}\\right)

        .. math::

            \\theta = \sqrt{\\frac{\sigma\mu}{4t}}
    """
    theta = np.sqrt((sigma*mu_0)/(4*t))
    const = -1/(mu_0*sigma*a**3)
    ta = theta*a
    eta = spec.erf(ta)
    t1 = 3*eta
    t2 = -2/(np.pi**0.5)*ta*(3+2*ta**2)*np.exp(-ta**2)
    dhzdt = const*(t1+t2)
    return mu_0*dhzdt


def Bz_horizontal_circular_loop_ColeCole(a, t, sigma):

    wt, tbase, omega_int = setFrequency(t)
    hz = Hz_horizontal_circular_loop(omega_int/2/np.pi, 1., a, sigma, 'secondary')
    # Treatment for inaccuracy in analytic solutions
    ind = omega_int < 0.2
    hz[ind] = 0.
    hzTD, f0 = transFilt(hz, wt, tbase, omega_int, t)
    return hzTD*mu_0


def dBzdt_horizontal_circular_loop_ColeCole(a, t, sigma):

    wt, tbase, omega_int = setFrequency(t)
    hz = Hz_horizontal_circular_loop(omega_int/2/np.pi, 1., a, sigma, 'secondary')
    # Treatment for inaccuracy in analytic solutions
    ind = omega_int < 0.2
    hz[ind] = 0.
    dhzdtTD = -transFiltImpulse(hz, wt, tbase, omega_int, t)

    return dhzdtTD*mu_0


def Bz_horizontal_circular_loop_VRM(a, z, h, t, dchi, tau1, tau2):

    mu0 = 4*np.pi*1e-7
    F = - (1/np.log(tau2/tau1)) * (spec.expi(t/tau2) + spec.expi(-t/tau1))
    B0 = (0.5*mu0*a**2) * (dchi/(2 + dchi)) * ((z + h)**2 + a**2)**-1.5
    return B0*F


def dBzdt_horizontal_circular_loop_VRM(a, z, h, t, dchi, tau1, tau2):

    mu0 = 4*np.pi*1e-7
    dFdt = (1/np.log(tau2/tau1)) * (np.exp(-t/tau1) - np.exp(-t/tau2)) / t
    B0 = (0.5*mu0*a**2) * (dchi/(2 + dchi)) * ((z + h)**2 + a**2)**-1.5
    return B0*dFdt

