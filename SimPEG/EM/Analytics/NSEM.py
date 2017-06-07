import numpy as _np
from scipy.constants import epsilon_0 as _epsilon_0
from scipy.constants import mu_0 as _mu_0
from SimPEG.EM.Utils.EMUtils import k as _k
from SimPEG.EM.Utils.EMUtils import omega as _omega

# Evaluate Impedance Z of a layer
_ImpZ = lambda f, mu, k: _omega(f)*mu/k

# Complex Cole-Cole Conductivity - EM utils
_PCC = lambda siginf, m, t, c, f: siginf*(1.-(m/(1.+(1j*_omega(f)*t)**c)))

# matrix P relating Up and Down components with E and H fields
_P = lambda z: _np.matrix([[1., 1, ], [-1./z, 1./z]], dtype='complex_')
_Pinv = lambda z: _np.matrix([[1., -z], [1., z]], dtype='complex_')/2.

# matrix T for transition of Up and Down components accross a layer
_T = lambda h, k: _np.matrix([[_np.exp(1j*k*h), 0.], [0., _np.exp(-1j*k*h)]], dtype='complex_')
_Tinv = lambda h, k: _np.matrix([[_np.exp(-1j*k*h), 0.], [0., _np.exp(1j*k*h)]], dtype='complex_')

# Propagate Up and Down component for a certain frequency & evaluate E and H field
def _Propagate(f, thickness, sig, chg, taux, c, mu_r, eps_r, n):

    if isinstance(sig, float):
        sigmodel = _np.r_[sig]
    else:
        sigmodel = sig

    if isinstance(eps_r, float):
        epsmodel = _np.ones_like(sigmodel)*eps_r
    else:
        epsmodel = eps_r

    if isinstance(mu_r, float):
        mumodel = _np.ones_like(sigmodel)*mu_r
    else:
        epsmodel = mu_r

    sigcm = _np.zeros_like(sigmodel, dtype='complex_')
    if chg == 0. or taux == 0. or c == 0.:
        sigcm = sigmodel
    else:
        for j in range(1, len(sigcm)):
            sigcm[j] = _PCC(sigmodel[j], chg[j], taux[j], c[j], f)

    sigcm = _np.append(_np.r_[0.], sigcm)
    mu = _np.append(_np.r_[1.], mumodel)*_mu_0
    eps = _np.append(_np.r_[1.], epsmodel)*_epsilon_0
    H = _np.append(_np.r_[1.2*(1e5)], thickness)

    K = _k(f, sigcm, mu, eps)
    Z = _ImpZ(f, mu, K)

    EH = _np.matrix(_np.zeros((2, n+1), dtype='complex_'), dtype='complex_')
    UD = _np.matrix(_np.zeros((2, n+1), dtype='complex_'), dtype='complex_')

    UD[1, -1] = 1.

    for i in range(-2, -(n+2), -1):

        UD[:, i] = _Tinv(H[i+1], K[i])*_Pinv(Z[i])*_P(Z[i+1])*UD[:, i+1]
        UD = UD/((_np.abs(UD[0, :]+UD[1, :])).max())

    for j in range(0, n+1):
        EH[:, j] = _np.matrix([[1., 1, ], [-1./Z[j], 1./Z[j]]])*UD[:, j]

    return UD, EH, Z, K

# Utils to compute the apparent impedance over a layered Earth Model
def AppImpedance_LayeredEarth(freq, thickness, sig, nlayer, format='Res-Phase', chg=0., tau=0., c=0., mu_r=1., eps_r=1.):
    '''
    This code compute the analytic response of a n-layered Earth to a plane wave (Magnetotellurics).
    The solution is first developed in Ward and Hohmann 1988.
    See also http://em.geosci.xyz/content/maxwell3_fdem/natural_sources/MT_N_layered_Earth.html

    freq: frequency, either float or numpy array
    thickness: thickness of the Earth layers in meters. float or numpy array
    sig: electric conductivity of the Earth layers in S/m. float or numpy array
    nlayer: number of Earth layer. Integer.
    format: Output format. 'Res-Phase' returns apparent resisitivity and Phase. 'Complex' returns the complex Impedance
    chg, tau, c: Cole-Cole Parameters for chargeable layers. numpy array
    mu_r: relative magnetic permeability. float or numpy array
    eps_r: relative dielectric permittivity. float or numpy array
    '''
    if isinstance(freq, float):
        F = _np.r_[freq]
    else:
        F = freq

    Res = _np.zeros_like(F)
    Phase = _np.zeros_like(F)
    App_ImpZ = _np.zeros_like(F, dtype='complex_')

    for i in range(0, len(F)):
        _, EH, _, _ = _Propagate(F[i], thickness, sig, chg, tau, c, mu_r, eps_r, nlayer)

        App_ImpZ[i] = EH[0, 1]/EH[1, 1]

        Res[i] = _np.abs(App_ImpZ[i])**2./(_mu_0*_omega(F[i]))
        Phase[i] = _np.angle(App_ImpZ[i], deg=True)

    if format == 'Res-Phase':
        return Res, Phase

    elif format == 'Complex':
        return App_ImpZ


def _run():

    nlayer = 2
    F = _np.r_[1e-5, 1e3]
    H = 200.
    sign = _np.r_[0.1, 1.]

    nlayer1 = 3
    F1 = 1e-3
    H1 = _np.r_[200., 50.]
    sign1 = _np.r_[0.01, 1., 0.1]
    fm = 'Complex'

    Res, Phase = AppImpedance_LayeredEarth(F, H, sign, nlayer)
    print(Res, Phase)
    appimp = AppImpedance_LayeredEarth(F1, H1, sign1, nlayer1, format=fm)
    print(appimp)

if __name__ == '__main__':
    _run()