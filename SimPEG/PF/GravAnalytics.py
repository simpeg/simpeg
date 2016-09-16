from __future__ import print_function
from scipy.constants import G
from SimPEG.Utils import mkvc
import numpy as np


def GravSphereFreeSpace(x, y, z, R, xc, yc, zc, rho):
    """
        Computing the induced response of magnetic sphere in free-space.
        >> Input
        x, y, z:   Observation locations
        R:     radius of the sphere
        xc, yc, zc: Location of the sphere
        rho: Density of sphere

        By convention, z-component is positive downward for a
        positive density anomaly

    """
    if (~np.size(x) == np.size(y) == np.size(z)):
        print("Specify same size of x, y, z")
        return

    unit_conv = 1e+8  # Unit conversion from SI to (mgal*g/cc)
    x = mkvc(x)
    y = mkvc(y)
    z = mkvc(z)

    nobs = len(x)

    M = R**3. * 4. / 3. * np.pi * rho

    rx = (x - xc)
    ry = (y - yc)
    rz = (zc - z)

    rvec = np.c_[rx, ry, rz]
    r = np.sqrt((rx)**2+(ry)**2+(rz)**2)

    g = -G*(1./r**2)*M * unit_conv

    gx = g * (rx / r)
    gy = g * (ry / r)
    gz = g * (rz / r)

    return gx, gy, gz
