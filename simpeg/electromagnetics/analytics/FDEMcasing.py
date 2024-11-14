import numpy as np

from scipy.constants import mu_0, epsilon_0
from simpeg.electromagnetics.utils import k


def getKc(freq, conductivity, a, b, permeability=mu_0, eps=epsilon_0):
    a = float(a)
    b = float(b)
    # return 1./(2*np.pi) * np.sqrt(b / a) * np.exp(-1j*k(freq,conductivity,permeability,eps)*(b-a))
    return np.sqrt(b / a) * np.exp(
        -1j * k(freq, conductivity, permeability, eps) * (b - a)
    )


def _r2(xyz):
    return np.sum(xyz**2, 1)


def _getCasingHertzMagDipole(
    srcloc,
    obsloc,
    freq,
    conductivity,
    a,
    b,
    permeability=(mu_0, mu_0, mu_0),
    eps=epsilon_0,
    moment=1.0,
):
    permeability = np.asarray(permeability)
    Kc1 = getKc(freq, conductivity[1], a, b, permeability[1], eps)

    nobs = obsloc.shape[0]
    dxyz = obsloc - np.c_[np.ones(nobs)] * np.r_[srcloc]

    r2 = _r2(dxyz[:, :2])
    sqrtr2z2 = np.sqrt(r2 + dxyz[:, 2] ** 2)
    k2 = k(freq, conductivity[2], permeability[2], eps)

    return Kc1 * moment / (4.0 * np.pi) * np.exp(-1j * k2 * sqrtr2z2) / sqrtr2z2


def _getCasingHertzMagDipoleDeriv_r(
    srcloc,
    obsloc,
    freq,
    conductivity,
    a,
    b,
    permeability=(mu_0, mu_0, mu_0),
    eps=epsilon_0,
    moment=1.0,
):
    permeability = np.asarray(permeability)
    HertzZ = _getCasingHertzMagDipole(
        srcloc, obsloc, freq, conductivity, a, b, permeability, eps, moment
    )

    nobs = obsloc.shape[0]
    dxyz = obsloc - np.c_[np.ones(nobs)] * np.r_[srcloc]

    r2 = _r2(dxyz[:, :2])
    sqrtr2z2 = np.sqrt(r2 + dxyz[:, 2] ** 2)
    k2 = k(freq, conductivity[2], permeability[2], eps)

    return -HertzZ * np.sqrt(r2) / sqrtr2z2 * (1j * k2 + 1.0 / sqrtr2z2)


def _getCasingHertzMagDipoleDeriv_z(
    srcloc,
    obsloc,
    freq,
    conductivity,
    a,
    b,
    permeability=(mu_0, mu_0, mu_0),
    eps=epsilon_0,
    moment=1.0,
):
    permeability = np.asarray(permeability)
    HertzZ = _getCasingHertzMagDipole(
        srcloc, obsloc, freq, conductivity, a, b, permeability, eps, moment
    )

    nobs = obsloc.shape[0]
    dxyz = obsloc - np.c_[np.ones(nobs)] * np.r_[srcloc]

    r2z2 = _r2(dxyz)
    sqrtr2z2 = np.sqrt(r2z2)
    k2 = k(freq, conductivity[2], permeability[2], eps)

    return -HertzZ * dxyz[:, 2] / sqrtr2z2 * (1j * k2 + 1.0 / sqrtr2z2)


def _getCasingHertzMagDipole2Deriv_z_r(
    srcloc,
    obsloc,
    freq,
    conductivity,
    a,
    b,
    permeability=(mu_0, mu_0, mu_0),
    eps=epsilon_0,
    moment=1.0,
):
    permeability = np.asarray(permeability)
    HertzZ = _getCasingHertzMagDipole(
        srcloc, obsloc, freq, conductivity, a, b, permeability, eps, moment
    )
    dHertzZdr = _getCasingHertzMagDipoleDeriv_r(
        srcloc, obsloc, freq, conductivity, a, b, permeability, eps, moment
    )

    nobs = obsloc.shape[0]
    dxyz = obsloc - np.c_[np.ones(nobs)] * np.r_[srcloc]

    r2 = _r2(dxyz[:, :2])
    r = np.sqrt(r2)
    z = dxyz[:, 2]
    sqrtr2z2 = np.sqrt(r2 + z**2)
    k2 = k(freq, conductivity[2], permeability[2], eps)

    return dHertzZdr * (-z / sqrtr2z2) * (1j * k2 + 1.0 / sqrtr2z2) + HertzZ * (
        z * r / sqrtr2z2**3
    ) * (1j * k2 + 2.0 / sqrtr2z2)


def _getCasingHertzMagDipole2Deriv_z_z(
    srcloc,
    obsloc,
    freq,
    conductivity,
    a,
    b,
    permeability=(mu_0, mu_0, mu_0),
    eps=epsilon_0,
    moment=1.0,
):
    permeability = np.asarray(permeability)
    HertzZ = _getCasingHertzMagDipole(
        srcloc, obsloc, freq, conductivity, a, b, permeability, eps, moment
    )
    dHertzZdz = _getCasingHertzMagDipoleDeriv_z(
        srcloc, obsloc, freq, conductivity, a, b, permeability, eps, moment
    )

    nobs = obsloc.shape[0]
    dxyz = obsloc - np.c_[np.ones(nobs)] * np.r_[srcloc]

    r2 = _r2(dxyz[:, :2])
    z = dxyz[:, 2]
    sqrtr2z2 = np.sqrt(r2 + z**2)
    k2 = k(freq, conductivity[2], permeability[2], eps)

    return (dHertzZdz * z + HertzZ) / sqrtr2z2 * (
        -1j * k2 - 1.0 / sqrtr2z2
    ) + HertzZ * z / sqrtr2z2**3 * (1j * k2 * z + 2.0 * z / sqrtr2z2)


def getCasingEphiMagDipole(
    srcloc,
    obsloc,
    freq,
    conductivity,
    a,
    b,
    permeability=(mu_0, mu_0, mu_0),
    eps=epsilon_0,
    moment=1.0,
):
    permeability = np.asarray(permeability)
    omega = 2 * np.pi * freq
    return (
        1j
        * omega
        * permeability
        * _getCasingHertzMagDipoleDeriv_r(
            srcloc, obsloc, freq, conductivity, a, b, permeability, eps, moment
        )
    )


def getCasingHrMagDipole(
    srcloc,
    obsloc,
    freq,
    conductivity,
    a,
    b,
    permeability=(mu_0, mu_0, mu_0),
    eps=epsilon_0,
    moment=1.0,
):
    permeability = np.asarray(permeability)
    return _getCasingHertzMagDipole2Deriv_z_r(
        srcloc, obsloc, freq, conductivity, a, b, permeability, eps, moment
    )


def getCasingHzMagDipole(
    srcloc,
    obsloc,
    freq,
    conductivity,
    a,
    b,
    permeability=(mu_0, mu_0, mu_0),
    eps=epsilon_0,
    moment=1.0,
):
    permeability = np.asarray(permeability)
    d2HertzZdz2 = _getCasingHertzMagDipole2Deriv_z_z(
        srcloc, obsloc, freq, conductivity, a, b, permeability, eps, moment
    )
    k2 = k(freq, conductivity[2], permeability[2], eps)
    HertzZ = _getCasingHertzMagDipole(
        srcloc, obsloc, freq, conductivity, a, b, permeability, eps, moment
    )
    return d2HertzZdz2 + k2**2 * HertzZ


def getCasingBrMagDipole(
    srcloc,
    obsloc,
    freq,
    conductivity,
    a,
    b,
    permeability=(mu_0, mu_0, mu_0),
    eps=epsilon_0,
    moment=1.0,
):
    permeability = np.asarray(permeability)
    return mu_0 * getCasingHrMagDipole(
        srcloc, obsloc, freq, conductivity, a, b, permeability, eps, moment
    )


def getCasingBzMagDipole(
    srcloc,
    obsloc,
    freq,
    conductivity,
    a,
    b,
    permeability=(mu_0, mu_0, mu_0),
    eps=epsilon_0,
    moment=1.0,
):
    permeability = np.asarray(permeability)
    return mu_0 * getCasingHzMagDipole(
        srcloc, obsloc, freq, conductivity, a, b, permeability, eps, moment
    )
