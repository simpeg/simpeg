import numpy as np
from scipy.constants import epsilon_0
from scipy.constants import mu_0
from SimPEG.EM.Utils.EMUtils import k
from SimPEG.EM.Utils.EMUtils import omega

__all__ = [
    'MT_LayeredEarth'
]


# Evaluate Impedance Z of a layer
_ImpZ = lambda f, mu, k: omega(f)*mu/k

# Complex Cole-Cole Conductivity - EM utils
_PCC = lambda siginf, m, t, c, f: siginf*(1.-(m/(1.+(1j*omega(f)*t)**c)))

# matrix P relating Up and Down components with E and H fields
_P = lambda z: np.matrix([[1., 1, ], [-1./z, 1./z]], dtype='complex_')
_Pinv = lambda z: np.matrix([[1., -z], [1., z]], dtype='complex_')/2.

# matrix T for transition of Up and Down components accross a layer
_T = lambda h, k: np.matrix([[np.exp(1j*k*h), 0.], [0., np.exp(-1j*k*h)]], dtype='complex_')
_Tinv = lambda h, k: np.matrix([[np.exp(-1j*k*h), 0.], [0., np.exp(1j*k*h)]], dtype='complex_')

# Propagate Up and Down component for a certain frequency & evaluate E and H field
def _Propagate(f, thickness, sig, chg, taux, c, mu_r, eps_r, n):

    if isinstance(sig, float):
        sigmodel = np.r_[sig]
    else:
        sigmodel = sig

    if isinstance(eps_r, float):
        epsmodel = np.ones_like(sigmodel)*eps_r
    else:
        epsmodel = eps_r

    if isinstance(mu_r, float):
        mumodel = np.ones_like(sigmodel)*mu_r
    else:
        epsmodel = mu_r

    sigcm = np.zeros_like(sigmodel, dtype='complex_')
    if chg == 0. or taux == 0. or c == 0.:
        sigcm = sigmodel
    else:
        for j in range(1, len(sigcm)):
            sigcm[j] = _PCC(sigmodel[j], chg[j], taux[j], c[j], f)

    sigcm = np.append(np.r_[0.], sigcm)
    mu = np.append(np.r_[1.], mumodel)*mu_0
    eps = np.append(np.r_[1.], epsmodel)*epsilon_0
    H = np.append(np.r_[1.2*(1e5)], thickness)

    K = k(f, sigcm, mu, eps)
    Z = _ImpZ(f, mu, K)

    EH = np.matrix(np.zeros((2, n+1), dtype='complex_'), dtype='complex_')
    UD = np.matrix(np.zeros((2, n+1), dtype='complex_'), dtype='complex_')

    UD[1, -1] = 1.

    for i in range(-2, -(n+2), -1):

        UD[:, i] = _Tinv(H[i+1], K[i])*_Pinv(Z[i])*_P(Z[i+1])*UD[:, i+1]
        UD = UD/((np.abs(UD[0, :]+UD[1, :])).max())

    for j in range(0, n+1):
        EH[:, j] = np.matrix([[1., 1, ], [-1./Z[j], 1./Z[j]]])*UD[:, j]

    return UD, EH, Z, K

# Utils to compute the apparent impedance over a layered Earth Model
def MT_LayeredEarth(freq, thickness, sig, nlayer, return_type='Res-Phase', chg=0., tau=0., c=0., mu_r=1., eps_r=1.):
    '''
    This code compute the analytic response of a n-layered Earth to a plane wave (Magnetotellurics).
    The solution is first developed in Ward and Hohmann 1988.
    See also http://em.geosci.xyz/content/maxwell3_fdem/natural_sources/MT_N_layered_Earth.html

    :param freq: the frequency at which we take the measurements
    :type freq: float or numpy.array
    :param thickness: thickness of the Earth layers in meters
    :type thickness: float or numpy.array
    :param sig: electric conductivity of the Earth layers in S/m
    :type sig: float or numpy.array
    :param int nlayer: number of Earth layer
    :param str return_type: Output return_type. 'Res-Phase' returns apparent resisitivity and Phase. 'Complex' returns the complex Impedance
    :param numpy.array chg: Cole-Cole Parameters for chargeable layers: chargeability
    :param numpy.array tau: Cole-Cole Parameters for chargeable layers: time decay constant
    :param numpy.array c: Cole-Cole Parameters for chargeable layers: geometric factor
    :param mu_r: relative magnetic permeability
    :type mu_r: float or numpy.array
    :param eps_r: relative dielectric permittivity
    :type eps_r: float or numpy.array
    '''
    if isinstance(freq, float):
        F = np.r_[freq]
    else:
        F = freq

    Res = np.zeros_like(F)
    Phase = np.zeros_like(F)
    App_ImpZ = np.zeros_like(F, dtype='complex_')

    for i in range(0, len(F)):
        _, EH, _, _ = _Propagate(F[i], thickness, sig, chg, tau, c, mu_r, eps_r, nlayer)

        App_ImpZ[i] = EH[0, 1]/EH[1, 1]

        Res[i] = np.abs(App_ImpZ[i])**2./(mu_0*omega(F[i]))
        Phase[i] = np.angle(App_ImpZ[i], deg=True)

    if return_type == 'Res-Phase':
        return Res, Phase

    elif return_type == 'Complex':
        return App_ImpZ


def _run():

    nlayer = 2
    F = np.r_[1e-5, 1e3]
    H = 200.
    sign = np.r_[0.1, 1.]

    nlayer1 = 3
    F1 = 1e-3
    H1 = np.r_[200., 50.]
    sign1 = np.r_[0.01, 1., 0.1]
    fm = 'Complex'

    Res, Phase = AppImpedance_LayeredEarth(F, H, sign, nlayer)
    print(Res, Phase)
    appimp = AppImpedance_LayeredEarth(F1, H1, sign1, nlayer1, return_type=fm)
    print(appimp)

if __name__ == '__main__':
    _run()
