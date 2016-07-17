from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest
from SimPEG import *
from SimPEG import MT

TOL = 1e-6

def appResPhs(freq,z):
    app_res = (old_div((old_div(1.,(8e-7*np.pi**2))),freq))*np.abs(z)**2
    app_phs = np.arctan2(-z.imag,z.real)*(old_div(180,np.pi))
    return app_res, app_phs

def appResNorm(sigmaHalf):
    nFreq = 26

    m1d = Mesh.TensorMesh([[(100,5,1.5),(100.,10),(100,5,1.5)]], x0=['C'])
    sigma = np.zeros(m1d.nC) + sigmaHalf
    sigma[m1d.gridCC[:]>200] = 1e-8

    # Calculate the analytic fields
    freqs = np.logspace(4,-4,nFreq)
    Z = []
    for freq in freqs:
        Ed, Eu, Hd, Hu = MT.Utils.getEHfields(m1d,sigma,freq,np.array([200]))
        Z.append(old_div((Ed + Eu),(Hd + Hu)))

    Zarr = np.concatenate(Z)

    app_r, app_p = appResPhs(freqs,Zarr)

    return old_div(np.linalg.norm(np.abs(app_r - old_div(np.ones(nFreq),sigmaHalf))), np.log10(sigmaHalf))


class TestAnalytics(unittest.TestCase):

    def setUp(self):
        pass
    def test_appRes2en1(self):self.assertLess(appResNorm(2e-1), TOL)
    def test_appRes2en2(self):self.assertLess(appResNorm(2e-2), TOL)
    def test_appRes2en3(self):self.assertLess(appResNorm(2e-3), TOL)
    def test_appRes2en4(self):self.assertLess(appResNorm(2e-4), TOL)
    def test_appRes2en5(self):self.assertLess(appResNorm(2e-5), TOL)
    def test_appRes2en6(self):self.assertLess(appResNorm(2e-6), TOL)




if __name__ == '__main__':
    unittest.main()
