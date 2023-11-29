import numpy as np
from scipy.constants import epsilon_0
from scipy.constants import mu_0
from SimPEG.electromagnetics.utils import k, omega

__all__ = ["MT_LayeredEarth"]


# Evaluate Impedance Z of a layer
def _ImpZ(f, mu, k):
    return omega(f) * mu / k


# Complex Cole-Cole Conductivity - EM utils
def _PCC(siginf, m, t, c, f):
    return siginf * (1.0 - (m / (1.0 + (1j * omega(f) * t) ** c)))


# matrix P relating Up and Down components with E and H fields
def _P(z):
    return np.array([[1.0, 1.0], [-1.0 / z, 1.0 / z]])


def _Pinv(z):
    return np.array([[1.0, -z], [1.0, z]]) / 2.0


# matrix T for transition of Up and Down components accross a layer
def _T(h, k):
    return np.array([[np.exp(1j * k * h), 0.0], [0.0, np.exp(-1j * k * h)]])


def _Tinv(h, k):
    return np.array([[np.exp(-1j * k * h), 0.0], [0.0, np.exp(1j * k * h)]])


# Propagate Up and Down component for a certain frequency & evaluate E and H field
def _Propagate(f, thickness, sig, chg, taux, c, mu_r, eps_r, n):
    if isinstance(eps_r, float):
        epsmodel = np.ones_like(sig) * eps_r
    else:
        epsmodel = eps_r

    if isinstance(mu_r, float):
        mumodel = np.ones_like(sig) * mu_r
    else:
        epsmodel = mu_r

    sigcm = np.zeros_like(sig, dtype="complex_")
    if chg == 0.0 or taux == 0.0 or c == 0.0:
        sigcm = sig
    else:
        for j in range(1, len(sigcm)):
            sigcm[j] = _PCC(sig[j], chg[j], taux[j], c[j], f)

    sigcm = np.append(np.r_[0.0], sigcm)
    mu = np.append(np.r_[1.0], mumodel) * mu_0
    eps = np.append(np.r_[1.0], epsmodel) * epsilon_0
    H = np.append(np.r_[1.2 * (1e5)], thickness)

    K = k(f, sigcm, mu, eps)
    Z = _ImpZ(f, mu, K)

    EH = np.zeros((2, n + 1), dtype="complex_")
    UD = np.zeros((2, n + 1), dtype="complex_")

    UD[1, -1] = 1.0

    for i in range(-2, -(n + 2), -1):
        UD[:, i] = _Tinv(H[i + 1], K[i]) @ _Pinv(Z[i]) @ _P(Z[i + 1]) @ UD[:, i + 1]
        UD = UD / ((np.abs(UD[0, :] + UD[1, :])).max())

    for j in range(0, n + 1):
        EH[:, j] = (
            np.array(
                [
                    [
                        1.0,
                        1,
                    ],
                    [-1.0 / Z[j], 1.0 / Z[j]],
                ]
            )
            @ UD[:, j]
        )

    return UD, EH, Z, K


# Utils to compute the apparent impedance over a layered Earth Model
def MT_LayeredEarth(
    freq,
    thickness,
    sig,
    return_type="Res-Phase",
    chg=0.0,
    tau=0.0,
    c=0.0,
    mu_r=1.0,
    eps_r=1.0,
):
    """
    This code compute the analytic response of a n-layered Earth to a plane wave (Magnetotellurics).
    All physical properties arrays convention describes the layers parameters from the top layer to the bottom layer.
    The solution is first developed in Ward and Hohmann 1988.
    See also https://em.geosci.xyz/content/maxwell3_fdem/natural_sources/MT_N_layered_Earth.html

    :param freq: the frequency at which we take the measurements
    :type freq: float or numpy.ndarray
    :param thickness: thickness of the Earth layers in meters, size is len(sig)-1. The last one is already considered infinite. For 1-layer Earth, thickness = None or 0.
    :type thickness: float or numpy.ndarray
    :param sig: electric conductivity of the Earth layers in S/m
    :type sig: float or numpy.ndarray
    :param str return_type: Output return_type. 'Res-Phase' returns apparent resisitivity and Phase. 'Impedance' returns the complex Impedance
    :param numpy.ndarray chg: Cole-Cole Parameters for chargeable layers: chargeability
    :param numpy.ndarray tau: Cole-Cole Parameters for chargeable layers: time decay constant
    :param numpy.ndarray c: Cole-Cole Parameters for chargeable layers: geometric factor
    :param mu_r: relative magnetic permeability
    :type mu_r: float or numpy.ndarray
    :param eps_r: relative dielectric permittivity
    :type eps_r: float or numpy.ndarray
    """

    if isinstance(freq, float):
        F = np.r_[freq]
    else:
        F = freq

    if isinstance(sig, float):
        sigmodel = np.r_[sig]
    else:
        sigmodel = sig

    if isinstance(thickness, float):
        if thickness == 0.0:
            thickmodel = np.empty(0)
        else:
            thickmodel = np.r_[thickness]
    elif thickness is None:
        thickmodel = np.empty(0)
    else:
        thickmodel = thickness

    # Count the number of layers
    nlayer = len(sigmodel)

    Res = np.zeros_like(F)
    Phase = np.zeros_like(F)
    App_ImpZ = np.zeros_like(F, dtype="complex_")

    for i in range(0, len(F)):
        _, EH, _, _ = _Propagate(
            F[i], thickmodel, sigmodel, chg, tau, c, mu_r, eps_r, nlayer
        )

        App_ImpZ[i] = EH[0, 1] / EH[1, 1]

        Res[i] = np.abs(App_ImpZ[i]) ** 2.0 / (mu_0 * omega(F[i]))
        Phase[i] = np.angle(App_ImpZ[i], deg=True)

    if return_type == "Res-Phase":
        return Res, Phase

    elif return_type == "Impedance":
        return App_ImpZ


def _run():
    # nlayer=1
    F0 = 1.0
    H0 = None
    H01 = 0.0
    sign0 = 0.1

    # nlayer = 2
    F1 = np.r_[1e-5, 1e3]
    H1 = 200.0
    sign1 = np.r_[0.1, 1.0]

    # nlayer1 = 3
    F2 = 1e-3
    H2 = np.r_[200.0, 50.0]
    sign2 = np.r_[0.01, 1.0, 0.1]
    fm = "Impedance"

    Res, Phase = MT_LayeredEarth(F0, H0, sign0)
    print(Res, Phase)
    Res, Phase = MT_LayeredEarth(F0, H01, sign0)
    print(Res, Phase)
    Res, Phase = MT_LayeredEarth(F1, H1, sign1)
    print(Res, Phase)
    appimp = MT_LayeredEarth(F2, H2, sign2, return_type=fm)
    print(appimp)


if __name__ == "__main__":
    _run()
