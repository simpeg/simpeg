import numpy as np
from SimPEG import Utils
from scipy.constants import mu_0, epsilon_0
from SimPEG.EM.Utils.EMUtils import k

def getKc(freq,sigma,a,b,mu=mu_0,eps=epsilon_0):
    a = float(a)
    b = float(b)
    # return 1./(2*np.pi) * np.sqrt(b / a) * np.exp(-1j*k(freq,sigma,mu,eps)*(b-a))
    return np.sqrt(b / a) * np.exp(-1j*k(freq,sigma,mu,eps)*(b-a))

def _r2(xyz):
    return np.sum(xyz**2,1)

def _getCasingHertzMagDipole(srcloc,obsloc,freq,sigma,a,b,mu=mu_0*np.ones(3),eps=epsilon_0,moment=1.):
    Kc1 = getKc(freq,sigma[1],a,b,mu[1],eps)

    nobs = obsloc.shape[0]
    dxyz = obsloc - np.c_[np.ones(nobs)]*np.r_[srcloc]

    r2 = _r2(dxyz[:,:2])
    sqrtr2z2 = np.sqrt(r2 + dxyz[:,2]**2)
    k2 = k(freq,sigma[2],mu[2],eps)

    return Kc1 * moment / (4.*np.pi) *np.exp(-1j*k2*sqrtr2z2) / sqrtr2z2


def _getCasingHertzMagDipoleDeriv_r(srcloc,obsloc,freq,sigma,a,b,mu=mu_0*np.ones(3),eps=epsilon_0,moment=1.):
    HertzZ = _getCasingHertzMagDipole(srcloc,obsloc,freq,sigma,a,b,mu,eps,moment)

    nobs = obsloc.shape[0]
    dxyz = obsloc - np.c_[np.ones(nobs)]*np.r_[srcloc]

    r2 = _r2(dxyz[:,:2])
    sqrtr2z2 = np.sqrt(r2 + dxyz[:,2]**2)
    k2 = k(freq,sigma[2],mu[2],eps)

    return -HertzZ * np.sqrt(r2) / sqrtr2z2 * (1j*k2 + 1./ sqrtr2z2)


def _getCasingHertzMagDipoleDeriv_z(srcloc,obsloc,freq,sigma,a,b,mu=mu_0*np.ones(3),eps=epsilon_0,moment=1.):
    HertzZ = _getCasingHertzMagDipole(srcloc,obsloc,freq,sigma,a,b,mu,eps,moment)

    nobs = obsloc.shape[0]
    dxyz = obsloc - np.c_[np.ones(nobs)]*np.r_[srcloc]

    r2z2 = _r2(dxyz)
    sqrtr2z2 = np.sqrt(r2z2)
    k2 = k(freq,sigma[2],mu[2],eps)

    return -HertzZ*dxyz[:,2] /sqrtr2z2 * (1j*k2 + 1./sqrtr2z2)

def _getCasingHertzMagDipole2Deriv_z_r(srcloc,obsloc,freq,sigma,a,b,mu=mu_0*np.ones(3),eps=epsilon_0,moment=1.):
    HertzZ = _getCasingHertzMagDipole(srcloc,obsloc,freq,sigma,a,b,mu,eps,moment)
    dHertzZdr = _getCasingHertzMagDipoleDeriv_r(srcloc,obsloc,freq,sigma,a,b,mu,eps,moment)

    nobs = obsloc.shape[0]
    dxyz = obsloc - np.c_[np.ones(nobs)]*np.r_[srcloc]

    r2 = _r2(dxyz[:,:2])
    r = np.sqrt(r2)
    z = dxyz[:,2]
    sqrtr2z2 = np.sqrt(r2 + z**2)
    k2 = k(freq,sigma[2],mu[2],eps)

    return dHertzZdr*(-z/sqrtr2z2)*(1j*k2+1./sqrtr2z2) + HertzZ*(z*r/sqrtr2z2**3)*(1j*k2 + 2./sqrtr2z2)

def _getCasingHertzMagDipole2Deriv_z_z(srcloc,obsloc,freq,sigma,a,b,mu=mu_0*np.ones(3),eps=epsilon_0,moment=1.):
    HertzZ = _getCasingHertzMagDipole(srcloc,obsloc,freq,sigma,a,b,mu,eps,moment)
    dHertzZdz = _getCasingHertzMagDipoleDeriv_z(srcloc,obsloc,freq,sigma,a,b,mu,eps,moment)

    nobs = obsloc.shape[0]
    dxyz = obsloc - np.c_[np.ones(nobs)]*np.r_[srcloc]

    r2 = _r2(dxyz[:,:2])
    r = np.sqrt(r2)
    z = dxyz[:,2]
    sqrtr2z2 = np.sqrt(r2 + z**2)
    k2 = k(freq,sigma[2],mu[2],eps)

    return (dHertzZdz*z + HertzZ)/sqrtr2z2*(-1j*k2 - 1./sqrtr2z2) + HertzZ*z/sqrtr2z2**3*(1j*k2*z + 2.*z/sqrtr2z2)

def getCasingEphiMagDipole(srcloc,obsloc,freq,sigma,a,b,mu=mu_0*np.ones(3),eps=epsilon_0,moment=1.):
    return 1j * omega(freq) * mu * _getCasingHertzMagDipoleDeriv_r(srcloc,obsloc,freq,sigma,a,b,mu,eps,moment)

def getCasingHrMagDipole(srcloc,obsloc,freq,sigma,a,b,mu=mu_0*np.ones(3),eps=epsilon_0,moment=1.):
    return _getCasingHertzMagDipole2Deriv_z_r(srcloc,obsloc,freq,sigma,a,b,mu,eps,moment)

def getCasingHzMagDipole(srcloc,obsloc,freq,sigma,a,b,mu=mu_0*np.ones(3),eps=epsilon_0,moment=1.):
    d2HertzZdz2 = _getCasingHertzMagDipole2Deriv_z_z(srcloc,obsloc,freq,sigma,a,b,mu,eps,moment)
    k2 = k(freq,sigma[2],mu[2],eps)
    HertzZ = _getCasingHertzMagDipole(srcloc,obsloc,freq,sigma,a,b,mu,eps,moment)
    return d2HertzZdz2 + k2**2 * HertzZ

def getCasingBrMagDipole(srcloc,obsloc,freq,sigma,a,b,mu=mu_0*np.ones(3),eps=epsilon_0,moment=1.):
    return mu_0 * getCasingHrMagDipole(srcloc,obsloc,freq,sigma,a,b,mu,eps,moment)

def getCasingBzMagDipole(srcloc,obsloc,freq,sigma,a,b,mu=mu_0*np.ones(3),eps=epsilon_0,moment=1.):
    return mu_0 * getCasingHzMagDipole(srcloc,obsloc,freq,sigma,a,b,mu,eps,moment)
