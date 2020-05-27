from __future__ import division
import numpy as np
from scipy.constants import mu_0, pi, epsilon_0
from scipy.special import erf
from SimPEG import utils
import warnings


def hzAnalyticDipoleF(r, freq, sigma, secondary=True, mu=mu_0):
    """
    The analytical expression is given in Equation 4.56 in Ward and Hohmann,
    1988, and the example reproduces their Figure 4.2.


    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from SimPEG import electromagnetics as EM
        freq = np.logspace(-1, 5, 301)
        test = EM.analytics.hzAnalyticDipoleF(
                100, freq, 0.01, secondary=False)
        plt.loglog(freq, test.real, 'C0-', label='Real')
        plt.loglog(freq, -test.real, 'C0--')
        plt.loglog(freq, test.imag, 'C1-', label='Imaginary')
        plt.loglog(freq, -test.imag, 'C1--')
        plt.title('Response at $r=100$ m')
        plt.xlim([1e-1, 1e5])
        plt.ylim([1e-12, 1e-6])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('$H_z$ (A/m)')
        plt.legend(loc=6)
        plt.show()


    **Reference**

    - Ward, S. H., and G. W. Hohmann, 1988, Electromagnetic theory for
      geophysical applications, Chapter 4 of Electromagnetic Methods in Applied
      Geophysics: SEG, Investigations in Geophysics No. 3, 130--311; DOI:
      `10.1190/1.9781560802631.ch4
      <https://doi.org/10.1190/1.9781560802631.ch4>`_.

    """
    r = np.abs(r)
    k = np.sqrt(-1j * 2.0 * np.pi * freq * mu * sigma)

    m = 1
    front = m / (2.0 * np.pi * (k ** 2) * (r ** 5))
    back = 9 - (
        9 + 9j * k * r - 4 * (k ** 2) * (r ** 2) - 1j * (k ** 3) * (r ** 3)
    ) * np.exp(-1j * k * r)
    hz = front * back

    if secondary:
        hp = -1 / (4 * np.pi * r ** 3)
        hz = hz - hp

    if hz.ndim == 1:
        hz = utils.mkvc(hz, 2)

    return hz


def MagneticDipoleWholeSpace(
    XYZ, srcLoc, sig, f, moment, fieldType="b", mu_r=1, eps_r=1, **kwargs
):
    """
    Analytical solution for a dipole in a whole-space.

    The analytical expression is given in Equation 2.57 in Ward and Hohmann,
    1988, and the example reproduces their Figure 2.2.

    TODOs:
        - set it up to instead take a mesh & survey
        - add divide by zero safety


    .. plot::

        import numpy as np
        from SimPEG import electromagnetics as EM
        import matplotlib.pyplot as plt
        from scipy.constants import mu_0
        freqs = np.logspace(-2, 5, 301)
        Bx, By, Bz = EM.analytics.FDEM.MagneticDipoleWholeSpace(
                [0, 100, 0], [0, 0, 0], 1e-2, freqs, moment='Z')
        plt.figure()
        plt.loglog(freqs, Bz.real/mu_0, 'C0', label='Real')
        plt.loglog(freqs, -Bz.real/mu_0, 'C0--')
        plt.loglog(freqs, Bz.imag/mu_0, 'C1', label='Imaginary')
        plt.loglog(freqs, -Bz.imag/mu_0, 'C1--')
        plt.legend()
        plt.xlim([1e-2, 1e5])
        plt.ylim([1e-13, 1e-6])
        plt.show()

    **Reference**

    - Ward, S. H., and G. W. Hohmann, 1988, Electromagnetic theory for
      geophysical applications, Chapter 4 of Electromagnetic Methods in Applied
      Geophysics: SEG, Investigations in Geophysics No. 3, 130--311; DOI:
      `10.1190/1.9781560802631.ch4
      <https://doi.org/10.1190/1.9781560802631.ch4>`_.

    """

    orient = kwargs.pop("orientation", None)
    if orient is not None:
        warnings.warn(
            "orientation kwarg has been deprecated and will be removed"
            " in SimPEG version 0.15.0, please use the moment argument",
            DeprecationWarning,
        )
        magnitude = moment
        moment = orient
    else:
        magnitude = 1
    mu = kwargs.pop("mu", None)
    if mu is not None:
        warnings.warn(
            "mu kwarg has been deprecated and will be removed"
            " in SimPEG version 0.15.0, please use the mu_r argument.",
            DeprecationWarning,
        )
        mu_r = mu / mu_0

    mu = mu_0 * mu_r
    eps = epsilon_0 * eps_r
    w = 2 * np.pi * f

    if isinstance(moment, str):
        if moment == "X":
            mx, my, mz = 1.0, 0.0, 0.0
        elif moment == "Y":
            mx, my, mz = 0.0, 1.0, 0.0
        elif moment == "Z":
            mx, my, mz = 0.0, 0.0, 1.0
        else:
            raise NotImplementedError("String type for moment not recognized")
        mx, my, mz = mx * magnitude, my * magnitude, mz * magnitude
    else:
        mx, my, mz = moment[0], moment[1], moment[2]

    XYZ = utils.asArray_N_x_Dim(XYZ, 3)

    dx = XYZ[:, 0] - srcLoc[0]
    dy = XYZ[:, 1] - srcLoc[1]
    dz = XYZ[:, 2] - srcLoc[2]

    r = np.sqrt(dx ** 2.0 + dy ** 2.0 + dz ** 2.0)
    k = np.sqrt(-1j * w * mu * sig + w ** 2 * mu * eps)
    kr = k * r

    if fieldType in ["h", "b"]:
        front = 1 / (4.0 * pi * r ** 3.0) * np.exp(-1j * kr)
        mid = -(kr ** 2.0) + 3.0 * 1j * kr + 3.0

        Fx = front * (
            mx * ((dx / r) ** 2.0 * mid + (kr ** 2.0 - 1j * kr - 1.0))
            + my * ((dy * dx / r ** 2.0) * mid)
            + mz * ((dx * dz / r ** 2.0) * mid)
        )

        Fy = front * (
            mx * ((dx * dy / r ** 2.0) * mid)
            + my * ((dy / r) ** 2.0 * mid + (kr ** 2.0 - 1j * kr - 1.0))
            + mz * ((dy * dz / r ** 2.0) * mid)
        )

        Fz = front * (
            mx * ((dx * dz / r ** 2.0) * mid)
            + my * ((dy * dz / r ** 2.0) * mid)
            + mz * ((dz / r) ** 2.0 * mid + (kr ** 2.0 - 1j * kr - 1.0))
        )

        if fieldType == "b":
            Fx, Fy, Fz = mu * Fx, mu * Fy, mu * Fz

    elif fieldType == "e":

        front = 1j * w * mu * (1 + 1j * kr) / (4.0 * pi * r ** 3.0) * np.exp(-1j * kr)

        Fx = front * (my * (dz / r) + mz * (-dy / r))

        Fy = front * (mx * (-dz / r) + mz * (dx / r))

        Fz = front * (mx * (dy / r) + my * (-dx / r))

    return Fx, Fy, Fz


def ElectricDipoleWholeSpace(
    XYZ, srcLoc, sig, f, moment="X", fieldType="e", mu_r=1, eps_r=1, **kwargs
):

    orient = kwargs.pop("orientation", None)
    if orient is not None:
        warnings.warn(
            "orientation kwarg has been deprecated and will be removed"
            " in SimPEG version 0.15.0, please use the moment argument.",
            DeprecationWarning,
        )
        moment = orient
    mu = kwargs.pop("mu", None)
    if mu is not None:
        warnings.warn(
            "mu kwarg has been deprecated and will be removed"
            " in SimPEG version 0.15.0, please use the mu_r argument."
        )
        mu_r = mu / mu_0
    cur = kwargs.pop("current", None)
    if cur is not None:
        warnings.warn(
            "current kwarg has been deprecated and will be removed"
            " in SimPEG version 0.15.0, please use the moment argument."
        )
        magnitude = cur
    else:
        magnitude = 1
    length = kwargs.pop("length", None)
    if length is not None:
        warnings.warn(
            "length kwarg has been deprecated and will be removed"
            " in SimPEG version 0.15.0, please use the moment argument."
        )
        magnitude *= length

    mu = mu_0 * mu_r
    eps = epsilon_0 * eps_r
    w = 2 * np.pi * f

    if isinstance(moment, str):
        if moment.upper() == "X":
            mx, my, mz = 1.0, 0.0, 0.0
        elif moment.upper() == "Y":
            mx, my, mz = 0.0, 1.0, 0.0
        elif moment.upper() == "Z":
            mx, my, mz = 0.0, 0.0, 1.0
        else:
            raise NotImplementedError("String type for moment not recognized")
        mx, my, mz = mx * magnitude, my * magnitude, mz * magnitude

    else:
        mx, my, mz = moment[0], moment[1], moment[2]

    XYZ = utils.asArray_N_x_Dim(XYZ, 3)

    dx = XYZ[:, 0] - srcLoc[0]
    dy = XYZ[:, 1] - srcLoc[1]
    dz = XYZ[:, 2] - srcLoc[2]

    r = np.sqrt(dx ** 2.0 + dy ** 2.0 + dz ** 2.0)
    k = np.sqrt(-1j * w * mu * sig + w ** 2 * mu * eps)
    kr = k * r

    if fieldType == "e":

        front = 1 / (4.0 * np.pi * sig * r ** 3) * np.exp(-1j * k * r)
        mid = -(k ** 2) * r ** 2 + 3 * 1j * k * r + 3

        Fx = front * (
            mx * ((dx ** 2 / r ** 2) * mid + (k ** 2 * r ** 2 - 1j * k * r - 1.0))
            + my * (dy * dx / r ** 2) * mid
            + mz * (dz * dx / r ** 2) * mid
        )

        Fy = front * (
            mx * (dx * dy / r ** 2) * mid
            + my * ((dy ** 2 / r ** 2) * mid + (k ** 2 * r ** 2 - 1j * k * r - 1.0))
            + mz * (dz * dy / r ** 2) * mid
        )

        Fz = front * (
            mx * (dx * dz / r ** 2) * mid
            + my * (dy * dz / r ** 2) * mid
            + mz * ((dz ** 2 / r ** 2) * mid + (k ** 2 * r ** 2 - 1j * k * r - 1.0))
        )

    elif fieldType in ["h", "b"]:

        front = (1 + 1j * kr) / (4.0 * np.pi * r ** 2) * np.exp(-1j * k * r)

        Fx = front * (my * (dz / r) + mz * (-dy / r))

        Fy = front * (mx * (-dz / r) + mz * (dx / r))

        Fz = front * (mx * (dy / r) + my * (-dx / r))

        if fieldType == "b":
            Fx, Fy, Fz = mu * Fx, mu * Fy, mu * Fz

    return Fx, Fy, Fz
