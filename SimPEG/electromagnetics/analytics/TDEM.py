from __future__ import division
import numpy as np
from scipy.constants import mu_0, pi
from scipy.special import erf
from SimPEG import utils


def hzAnalyticDipoleT(r, t, sigma):
    theta = np.sqrt((sigma * mu_0) / (4 * t))
    tr = theta * r
    etr = erf(tr)
    t1 = (9 / (2 * tr ** 2) - 1) * etr
    t2 = (1 / np.sqrt(pi)) * (9 / tr + 4 * tr) * np.exp(-(tr ** 2))
    hz = (t1 - t2) / (4 * pi * r ** 3)
    return hz


def hzAnalyticCentLoopT(a, t, sigma):
    theta = np.sqrt((sigma * mu_0) / (4 * t))
    ta = theta * a
    eta = erf(ta)
    t1 = (3 / (np.sqrt(pi) * ta)) * np.exp(-(ta ** 2))
    t2 = (1 - (3 / (2 * ta ** 2))) * eta
    hz = (t1 + t2) / (2 * a)
    return hz


def TransientMagneticDipoleWholeSpace(
    XYZ, srcLoc, sig, t, moment, fieldType="h", mu_r=1
):
    """
    Analytical solution for a dipole in a whole-space.

    """

    mu = 4 * np.pi * 1e-7 * mu_r

    if isinstance(moment, str):
        if moment == "X":
            mx, my, mz = 1.0, 0.0, 0.0
        elif moment == "Y":
            mx, my, mz = 0.0, 1.0, 0.0
        elif moment == "Z":
            mx, my, mz = 0.0, 0.0, 1.0
        else:
            raise NotImplementedError("String type for moment not recognized")

    else:
        mx, my, mz = moment[0], moment[1], moment[2]

    XYZ = utils.asArray_N_x_Dim(XYZ, 3)

    dx = XYZ[:, 0] - srcLoc[0]
    dy = XYZ[:, 1] - srcLoc[1]
    dz = XYZ[:, 2] - srcLoc[2]

    r = np.sqrt(dx ** 2.0 + dy ** 2.0 + dz ** 2.0)
    k = np.sqrt(mu * sig / (4 * t))
    kr = k * r

    if fieldType == "h":
        front = 1 / (4.0 * pi * r ** 3.0)
        mid = 3 * erf(kr) - (4 * kr ** 3 + 6 * kr) * np.exp(-(kr ** 2)) / np.sqrt(pi)
        end = -erf(kr) + (4 * kr ** 3 + 2 * kr) * np.exp(-(kr ** 2)) / np.sqrt(pi)

        Fx = front * (
            mx * ((dx / r) ** 2.0 * mid + end)
            + my * ((dy * dx / r ** 2.0) * mid)
            + mz * ((dx * dz / r ** 2.0) * mid)
        )

        Fy = front * (
            mx * ((dx * dy / r ** 2.0) * mid)
            + my * ((dy / r) ** 2.0 * mid + end)
            + mz * ((dy * dz / r ** 2.0) * mid)
        )

        Fz = front * (
            mx * ((dx * dz / r ** 2.0) * mid)
            + my * ((dy * dz / r ** 2.0) * mid)
            + mz * ((dz / r) ** 2.0 * mid + end)
        )

    elif fieldType == "dhdt":
        front = (4 * k ** 5 / (pi ** 1.5 * mu * sig)) * np.exp(-(kr ** 2))
        mid = kr ** 2
        end = 1 - kr ** 2

        Fx = front * (
            mx * ((dx / r) ** 2.0 * mid + end)
            + my * ((dy * dx / r ** 2.0) * mid)
            + mz * ((dx * dz / r ** 2.0) * mid)
        )

        Fy = front * (
            mx * ((dx * dy / r ** 2.0) * mid)
            + my * ((dy / r) ** 2.0 * mid + end)
            + mz * ((dy * dz / r ** 2.0) * mid)
        )

        Fz = front * (
            mx * ((dx * dz / r ** 2.0) * mid)
            + my * ((dy * dz / r ** 2.0) * mid)
            + mz * ((dz / r) ** 2.0 * mid + end)
        )

    elif fieldType == "e":

        front = (2 * k ** 5 / (pi ** 1.5 * sig)) * np.exp(-(kr ** 2))

        Fx = front * (my * (-dz / r) + mz * (dy / r))

        Fy = front * (mx * (dz / r) + mz * (-dx / r))

        Fz = front * (mx * (-dy / r) + my * (dx / r))

    return Fx, Fy, Fz


def TransientElectricDipoleWholeSpace(
    XYZ, srcLoc, sig, t, moment, fieldType="h", mu_r=1
):

    mu = 4 * np.pi * 1e-7 * mu_r

    if isinstance(moment, str):
        if moment.upper() == "X":
            mx, my, mz = 1.0, 0.0, 0.0
        elif moment.upper() == "Y":
            mx, my, mz = 0.0, 1.0, 0.0
        elif moment.upper() == "Z":
            mx, my, mz = 0.0, 0.0, 1.0
        else:
            raise NotImplementedError("String type for moment not recognized")

    else:
        mx, my, mz = moment[0], moment[1], moment[2]

    XYZ = utils.asArray_N_x_Dim(XYZ, 3)

    dx = XYZ[:, 0] - srcLoc[0]
    dy = XYZ[:, 1] - srcLoc[1]
    dz = XYZ[:, 2] - srcLoc[2]

    r = np.sqrt(dx ** 2.0 + dy ** 2.0 + dz ** 2.0)
    k = np.sqrt(mu * sig / (4 * t))
    kr = k * r

    if fieldType == "e":

        front = 1 / (4.0 * np.pi * sig * r ** 3)
        mid = 3 * erf(kr) - (4 * kr ** 3 + 6 * kr) * np.exp(-(kr ** 2)) / np.sqrt(pi)
        end = -erf(kr) + (4 * kr ** 3 + 2 * kr) * np.exp(-(kr ** 2)) / np.sqrt(pi)

        Fx = front * (
            mx * ((dx ** 2 / r ** 2) * mid + end)
            + my * (dy * dx / r ** 2) * mid
            + mz * (dz * dx / r ** 2) * mid
        )

        Fy = front * (
            mx * (dx * dy / r ** 2) * mid
            + my * ((dy ** 2 / r ** 2) * mid + end)
            + mz * (dz * dy / r ** 2) * mid
        )

        Fz = front * (
            mx * (dx * dz / r ** 2) * mid
            + my * (dy * dz / r ** 2) * mid
            + mz * ((dz ** 2 / r ** 2) * mid + end)
        )

    elif fieldType == "h":

        front = (1 / (4.0 * pi * r ** 3)) * (
            erf(kr) - 2 * kr * np.exp(-(kr ** 2)) / np.sqrt(pi)
        )

        Fx = front * (my * (-dz / r) + mz * (dy / r))

        Fy = front * (mx * (dz / r) + mz * (-dx / r))

        Fz = front * (mx * (-dy / r) + my * (dx / r))

    elif fieldType == "dhdt":

        front = -(2 * k ** 5 / (pi ** 1.5 * mu * sig)) * np.exp(-(kr ** 2))

        Fx = front * (my * (-dz / r) + mz * (dy / r))

        Fy = front * (mx * (dz / r) + mz * (-dx / r))

        Fz = front * (mx * (-dy / r) + my * (dx / r))

    return Fx, Fy, Fz
