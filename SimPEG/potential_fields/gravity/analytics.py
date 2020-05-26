from __future__ import print_function
from scipy.constants import G
from SimPEG.utils import mkvc
import numpy as np


def GravSphereFreeSpace(x, y, z, R, xc, yc, zc, rho):
    """
        Computing the gravity response of a sphere in free-space.
        >> Input
        x, y, z:   Observation locations
        R:     radius of the sphere
        xc, yc, zc: Location of the sphere
        rho: Density of sphere

    """
    if ~np.size(x) == np.size(y) == np.size(z):
        print("Specify same size of x, y, z")
        return

    unit_conv = 1e8  # Unit conversion from SI to (mgal*g/cc)
    x = mkvc(x)
    y = mkvc(y)
    z = mkvc(z)

    rx = x - xc
    ry = y - yc
    rz = z - zc

    r = np.sqrt((rx) ** 2 + (ry) ** 2 + (rz) ** 2)

    M = np.empty_like(x)  # create a vector of "Ms" if the point is outide
    M[r >= R] = R ** 3 * 4.0 / 3.0 * np.pi * rho  # outside points
    M[r < R] = r[r < R] ** 3 * 4.0 / 3.0 * np.pi * rho  # inside points

    g = -G * (1.0 / r ** 2) * M * unit_conv

    gx = g * (rx / r)
    gy = g * (ry / r)
    gz = g * (rz / r)

    return gx, gy, gz


def GravityGradientSphereFreeSpace(x, y, z, R, xc, yc, zc, rho):
    """
        Computing the induced response of magnetic sphere in free-space.
        >> Input
        x, y, z:   Observation locations
        R:     radius of the sphere
        xc, yc, zc: Location of the sphere
        rho: Density of sphere

    """
    if ~np.size(x) == np.size(y) == np.size(z):
        print("Specify same size of x, y, z")
        return

    unit_conv = 1e12  # Unit conversion from SI to (Eotvos)
    x = mkvc(x)
    y = mkvc(y)
    z = mkvc(z)

    rx = x - xc
    ry = y - yc
    rz = z - zc
    rx2 = rx * rx
    ry2 = ry * ry
    rz2 = rz * rz

    r = np.sqrt(rx2 + ry2 + rz2)
    bot = r * r * r * r * r

    M = np.empty_like(x)  # create a vector of "Ms" if the point is outide
    M[r >= R] = R ** 3 * 4.0 / 3.0 * np.pi * rho  # outside points
    M[r < R] = r[r < R] ** 3 * 4.0 / 3.0 * np.pi * rho  # inside points

    g = G * (1.0 / bot) * M * unit_conv

    gxx = g * (2 * rx2 - ry2 - rz2)
    gyy = g * (2 * ry2 - rx2 - rz2)
    gzz = -gxx - gyy
    gxy = g * (3 * rx * ry)
    gxz = g * (3 * rx * rz)
    gyz = g * (3 * ry * rz)

    return gxx, gxy, gxz, gyy, gyz, gzz
