import numpy as np
from scipy import special


def DCAnalytic_Pole_Dipole(txloc, rxlocs, sigma, current=1.,
                           earth_type="wholespace"):
    """
    Analytic solution for electric potential from a postive pole Tx, measured
    using a dipole Rx

    :param array txloc: a xyz location of A (+) electrode (np.r_[xa, ya, za])
    :param list rxlocs: xyz locations of M (+) and N (-) electrodes [M, N]

        e.g.
        rxlocs = [M, N]
        M: xyz locations of M (+) electrode (np.c_[xmlocs, ymlocs, zmlocs])
        N: xyz locations of N (-) electrode (np.c_[xnlocs, ynlocs, znlocs])

    :param float or complex sigma: values of conductivity
    :param string earth_type: values of conductivity ("wholsespace" or "halfspace")
    """
    A = txloc

    M = rxlocs[0]
    N = rxlocs[1]

    rAM = np.sqrt((M[:, 0]-A[0])**2 + (M[:, 1]-A[1])**2 + (M[:, 2]-A[2])**2)
    rAN = np.sqrt((N[:, 0]-A[0])**2 + (N[:, 1]-A[1])**2 + (N[:, 2]-A[2])**2)

    frontFactor = current/(4*np.pi*sigma)

    phiM = frontFactor*(1/rAM)
    phiN = frontFactor*(1/rAN)
    phi = phiM - phiN

    if earth_type == "halfspace":
        phi *= 2

    return phi

def DCAnalytic_Dipole_Pole(txlocs, rxlocs, sigma, current=1., earth_type="wholespace"):
    """
    Analytic solution for electric potential from a dipole source, measured
    using a pole Rx

    :param array txlocs: xyz location of A (+)  and B (-) electrodes [np.r_[xa, ya, za], np.r_[xb, yb, zb]]
    :param list rxlocs: a xyz location of M (+) electrode (np.r_[xm, ym, zm])

    :param float or complex sigma: values of conductivity
    :param float current: input current of Tx in [A]
    :param string earth_type: values of conductivity ("wholsespace" or "halfspace")
    """

    A = txlocs[0]
    B = txlocs[1]

    M = rxlocs

    rAM = np.sqrt((M[:, 0]-A[0])**2 + (M[:, 1]-A[1])**2 + (M[:, 2]-A[2])**2)
    rBM = np.sqrt((M[:, 0]-B[0])**2 + (M[:, 1]-B[1])**2 + (M[:, 2]-B[2])**2)

    frontFactor = current/(4*np.pi*sigma)

    phiM = frontFactor*(1/rAM - 1/rBM)
    phi = phiM

    if earth_type == "halfspace":
        phi *= 2

    return phi


def DCAnalytic_Pole_Pole(txloc, rxloc, sigma, current=1.,
                         earth_type="wholespace"):
    """
    Analytic solution for electric potential from a postive pole Tx,
    measured using a pole Rx

    :param array txloc: xyz location of A (+) electrode (np.r_[xa, ya, za])
    :param list rxlocs: xyz locations of M (+) electrode (np.r_[xm, ym, zm])

    :param float or complex sigma: values of conductivity
    :param string earth_type: values of conductivity ("wholsespace" or "halfspace")

    """
    A = txloc
    M = rxloc

    rAM = np.sqrt((M[:, 0]-A[0])**2 + (M[:, 1]-A[1])**2 + (M[:, 2]-A[2])**2)

    frontFactor = current/(4*np.pi*sigma)

    phi = frontFactor*(1/rAM)

    if earth_type == "halfspace":
        phi *= 2

    return phi


def DCAnalytic_Dipole_Dipole(txlocs, rxlocs, sigma, current=1., earth_type="wholespace"):
    """
    Analytic solution for electric potential from a dipole source

    :param array txlocs: xyz location of A (+)  and B (-) electrodes [np.r_[xa, ya, za], np.r_[xb, yb, zb]]
    :param list rxlocs: xyz locations of M (+) and N (-) electrodes [M, N]

    .. code::

        rxlocs = [M, N]
        M: xyz locations of M (+) electrode (np.c_[xmlocs, ymlocs, zmlocs])
        N: xyz locations of N (-) electrode (np.c_[xnlocs, ynlocs, znlocs])

    :param float or complex sigma: values of conductivity
    :param float current: input current of Tx in [A]
    :param string earth_type: values of conductivity ("wholsespace" or "halfspace")
    """

    A = txlocs[0]
    B = txlocs[1]

    M = rxlocs[0]
    N = rxlocs[1]

    rAM = np.sqrt((M[:, 0]-A[0])**2 + (M[:, 1]-A[1])**2 + (M[:, 2]-A[2])**2)
    rAN = np.sqrt((N[:, 0]-A[0])**2 + (N[:, 1]-A[1])**2 + (N[:, 2]-A[2])**2)
    rBM = np.sqrt((M[:, 0]-B[0])**2 + (M[:, 1]-B[1])**2 + (M[:, 2]-B[2])**2)
    rBN = np.sqrt((N[:, 0]-B[0])**2 + (N[:, 1]-B[1])**2 + (N[:, 2]-B[2])**2)

    frontFactor = current/(4*np.pi*sigma)

    phiM = frontFactor*(1/rAM - 1/rBM)
    phiN = frontFactor*(1/rAN - 1/rBN)
    phi = phiM - phiN

    if earth_type == "halfspace":
        phi *= 2

    return phi


def DCAnalyticSphere(txloc, rxloc, xc, radius, sigma, sigma1,
                     field_type="secondary", order=12, halfspace=False):
    """
    Parameters:

    :param array txloc: A (+) current electrode location (x, y, z)
    :param array xc: x center of depressed sphere
    :param array rxloc: M(+) electrode locations / (Nx3 array, # of electrodes)

    :param float radius: radius (float): radius of the sphere (m)
    :param float rho: resistivity of the background (ohm-m)
    :param float rho1: resistivity of the sphere
    :param string field_type: : "secondary", "total", "primary"
          (default="secondary")
          "secondary": secondary potential only due to sphere
          "primary": primary potential from the point source
          "total": "secondary"+"primary"
    :param float order: maximum order of Legendre polynomial (default=12)

    Written by Seogi Kang (skang@eos.ubc.ca)
    Ph.D. Candidate of University of British Columbia, Canada
    """

    Pleg = []
    # Compute Legendre Polynomial
    for i in range(order):
        Pleg.append(special.legendre(i, monic=0))

    rho = 1./sigma
    rho1 = 1./sigma1

    # Center of the sphere should be aligned in txloc in y-direction
    yc = txloc[1]
    xyz = np.c_[rxloc[:, 0]-xc, rxloc[:, 1]-yc, rxloc[:, 2]]
    r = np.sqrt((xyz**2).sum(axis=1))

    x0 = abs(txloc[0]-xc)

    costheta = xyz[:, 0]/r * (txloc[0]-xc)/x0
    # phi = np.zeros_like(r)
    R = (r**2+x0**2.-2.*r*x0*costheta)**0.5
    # primary potential in a whole space
    prim = rho*1./(4*np.pi*R)

    if field_type == "primary":
        return prim

    sphind = r < radius
    out = np.zeros_like(r)
    for n in range(order):
        An, Bn = AnBnfun(n, radius, x0, rho, rho1)
        dumout = An*r[~sphind]**(-n-1.)*Pleg[n](costheta[~sphind])
        out[~sphind] += dumout
        dumin = Bn*r[sphind]**(n)*Pleg[n](costheta[sphind])
        out[sphind] += dumin

    out[~sphind] += prim[~sphind]

    if halfspace:
        scale = 2
    else:
        scale = 1

    if field_type == "secondary":
        return scale*(out-prim)
    elif field_type == "total":
        return scale*out


def AnBnfun(n, radius, x0, rho, rho1, I=1.):
    const = I*rho/(4*np.pi)
    bunmo = n*rho + (n+1)*rho1
    An = const * radius**(2*n+1) / x0 ** (n+1.) * n * \
        (rho1-rho) / bunmo
    Bn = const * 1. / x0 ** (n+1.) * (2*n+1) * (rho1) / bunmo
    return An, Bn
