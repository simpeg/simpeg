import numpy as np
from scipy.constants import mu_0, pi
from scipy import special

def DCAnalyticHalf(txloc, rxlocs, sigma, earth_type="wholespace"):
    """
        Analytic solution for electric potential from a postive pole

        :param array txloc: a xyz location of A (+) electrode (np.r_[xa, ya, za])
        :param list rxlocs: xyz locations of M (+) and N (-) electrodes [M, N]

            e.g.
            rxlocs = [M, N]
            M: xyz locations of M (+) electrode (np.c_[xmlocs, ymlocs, zmlocs])
            N: xyz locations of N (-) electrode (np.c_[xnlocs, ynlocs, znlocs])

        :param float or complex sigma: values of conductivity
        :param string earth_type: values of conductivity ("wholsespace" or "halfspace")

    """
    M = rxlocs[0]
    N = rxlocs[1]

    rM = np.sqrt( (M[:,0]-txloc[0])**2 + (M[:,1]-txloc[1])**2 + (M[:,2]-txloc[1])**2 )
    rN = np.sqrt( (N[:,0]-txloc[0])**2 + (N[:,1]-txloc[1])**2 + (N[:,2]-txloc[1])**2 )

    phiM = 1./(4*np.pi*rM*sigma)
    phiN = 1./(4*np.pi*rN*sigma)
    phi = phiM - phiN

    if earth_type == "halfspace":
        phi *= 2

    return phi

deg2rad  = lambda deg: deg/180.*np.pi
rad2deg  = lambda rad: rad*180./np.pi

def DCAnalyticSphere(txloc, rxloc, xc, radius, sigma, sigma1, \
                 field_type = "secondary", order=12, halfspace=False):
# def DCSpherePointCurrent(txloc, rxloc, xc, radius, rho, rho1, \
#                  field_type = "secondary", order=12):
    """

        Parameters:

            :param array txloc: A (+) current electrode location (x,y,z)
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
    xyz = np.c_[rxloc[:,0]-xc, rxloc[:,1]-yc, rxloc[:,2]]
    r = np.sqrt( (xyz**2).sum(axis=1) )

    x0 = abs(txloc[0]-xc)

    costheta = xyz[:,0]/r * (txloc[0]-xc)/x0
    phi = np.zeros_like(r)
    R = (r**2+x0**2.-2.*r*x0*costheta)**0.5
    # primary potential in a whole space
    prim = rho*1./(4*np.pi*R)

    if field_type =="primary":
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
