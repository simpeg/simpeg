from SimPEG import Mesh, Maps, Utils, Tests
from SimPEG.EM import FDEM
import numpy as np
from scipy.constants import mu_0

import unittest

MuMax = 50.
TOL = 1e-10

class MuBaseTests(unittest.TestCase):

    def setUp(self):
        cs = 10.
        nc = 20.
        npad = 15.
        hx = [(cs, nc), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]

        self.freq = 1

        self.mesh = Mesh.CylMesh([hx, 1., hz], '0CC')

        self.m0 = MuMax*np.random.rand(self.mesh.nC)

        src = FDEM.Src.MagDipole(
            rxList=[], loc=np.r_[0., 0., 0.], freq=self.freq
        )

        self.prob = FDEM.Problem3D_e(
            self.mesh, sigma=self.m0, muMap=Maps.ChiMap(self.mesh)
        )

    def test_Aderiv(self, prbtype='e'):
        if prbtype == 'b':
            nu = mesh.nF
        elif prbtype == 'e':
            nu = self.mesh.nE
        v = np.random.rand(nu)

        def AderivFun(m):
            self.prob.model = m
            A = self.prob.getA(self.freq)
            Av = A*v
            self.prob.model = self. m0

            def ADeriv_dm(dm):
                return self.prob.getADeriv_mui(self.freq, v, dm)

            return Av, ADeriv_dm

        print('\n Testing ADeriv {}'.format(prbtype))
        Tests.checkDerivative(
            AderivFun, self.m0, plotIt=False,
            num=3, eps=1e-20
            )

    def test_Aadjoint(self, prbtype='e'):

        print('\n Testing A_adjoint')
        m = np.random.rand(self.prob.muMap.nP)
        u = np.random.rand(self.mesh.nE)
        v = np.random.rand(self.mesh.nE)

        self.prob.model = self.m0

        V1 = v.dot(self.prob.getADeriv_mui(self.freq, u, m))
        V2 = m.dot(self.prob.getADeriv_mui(self.freq, u, v, adjoint=True))
        passed = np.abs(V1-V2) < TOL * (np.abs(V1) + np.abs(V2))/2.
        print('AdjointTest {prbtype} {v1} {v2} {passed}'.format(
            prbtype=prbtype, v1=V1, v2=V2, passed=passed))
        self.assertTrue(passed)



if __name__ == '__main__':
    unittest.main()


