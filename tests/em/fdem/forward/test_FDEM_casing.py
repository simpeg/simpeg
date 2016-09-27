from SimPEG import Tests, Utils, np
import SimPEG.EM.Analytics.FDEMcasing as Casing
import unittest
from scipy.constants import mu_0


n = 50
freq = 1.
a = 5e-2
b = a + 1e-2
sigma = np.r_[10., 5.5e6, 1e-1]
mu = mu_0*np.r_[1.,100.,1.]
srcloc = np.r_[0., 0., 0.]
xobs = np.random.rand(n)+10.
yobs = np.zeros(n)
zobs = np.random.randn(n)
plotit = False

def CasingMagDipoleDeriv_r(x):
    obsloc = np.vstack([x, yobs, zobs]).T

    f = Casing._getCasingHertzMagDipole(srcloc,obsloc,freq,sigma,a,b,mu)
    g = Utils.sdiag(Casing._getCasingHertzMagDipoleDeriv_r(srcloc,obsloc,freq,sigma,a,b,mu))

    return  f, g

def CasingMagDipoleDeriv_z(z):
    obsloc = np.vstack([xobs, yobs, z]).T

    f = Casing._getCasingHertzMagDipole(srcloc,obsloc,freq,sigma,a,b,mu)
    g = Utils.sdiag(Casing._getCasingHertzMagDipoleDeriv_z(srcloc,obsloc,freq,sigma,a,b,mu))

    return  f, g

def CasingMagDipole2Deriv_z_r(x):
    obsloc = np.vstack([x, yobs, zobs]).T

    f = Casing._getCasingHertzMagDipoleDeriv_z(srcloc,obsloc,freq,sigma,a,b,mu)
    g = Utils.sdiag(Casing._getCasingHertzMagDipole2Deriv_z_r(srcloc,obsloc,freq,sigma,a,b,mu))

    return f, g

def CasingMagDipole2Deriv_z_z(z):
    obsloc = np.vstack([xobs, yobs, z]).T

    f = Casing._getCasingHertzMagDipoleDeriv_z(srcloc,obsloc,freq,sigma,a,b,mu)
    g = Utils.sdiag(Casing._getCasingHertzMagDipole2Deriv_z_z(srcloc,obsloc,freq,sigma,a,b,mu))

    return f, g



class Casing_DerivTest(unittest.TestCase):
    def test_derivs(self):
        Tests.checkDerivative(CasingMagDipoleDeriv_r,np.ones(n)*10+np.random.randn(n),plotIt=False)
        Tests.checkDerivative(CasingMagDipoleDeriv_z,np.random.randn(n),plotIt=False)
        Tests.checkDerivative(CasingMagDipole2Deriv_z_r,np.ones(n)*10+np.random.randn(n),plotIt=False)
        Tests.checkDerivative(CasingMagDipole2Deriv_z_z,np.random.randn(n),plotIt=False)


if __name__ == '__main__':
    unittest.main()
