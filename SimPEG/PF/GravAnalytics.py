from __future__ import print_function
from scipy.constants import G
from SimPEG.Utils import mkvc
import numpy as np


def GravSphereFreeSpace(x, y, z, R, xc, yc, zc, rho):
    """
        Computing the gravity response of a sphere in free-space.
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

    unit_conv_g = 1e+8  # Unit conversion from SI to (mgal*g/cc)
    unit_conv_gg = 1e+12  # Unit conversion from SI to (mgal*g/cc)

    x = mkvc(x)
    y = mkvc(y)
    z = mkvc(z)

    M = R**3. * 4. / 3. * np.pi * rho

    dx = (x - xc)
    dy = (y - yc)
    dz = (z - zc)

    r = np.sqrt((dx)**2.+(dy)**2.+(dz)**2.)

    g = -G*(1./r**2.)*M * unit_conv_g

    gx = g * (dx / r)
    gy = g * (dy / r)
    gz = -g * (dz / r)

    gxx = -G * M * unit_conv_gg * (r**2. - 3.*dx**2.)/r**5.
    gxy = -G * M * unit_conv_gg * (-3.*dx*dy)/r**5.
    gxz = -G * M * unit_conv_gg * (-3.*dx*dz)/r**5.

    gyy = -G * M * unit_conv_gg * (r**2. - 3.*dy**2.)/r**5.
    gyz = -G * M * unit_conv_gg * (-3.*dy*dz)/r**5.

    gzz = -G * M * unit_conv_gg * (r**2. - 3.*dz**2.)/r**5.

    return {'gx': gx, 'gy': gy, 'gz': gz,
            'gxx': gxx, 'gxy': gxy, 'gxz': gxz,
            'gyy': gyy, 'gyz': gyz, 'gzz': gzz}
