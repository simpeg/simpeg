from SimPEG import Mesh, Utils, Tests
from SimPEG.EM import FDEM
import numpy as np
from scipy.constants import mu_0

import unittest


class MuBaseTests(unittest.TestCase):

    def setUp(self):
        cs = 10.
        nc = 20.
        npad = 15.
        hx = [(cs, nc), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]

        self.freq = 1

        self.mesh = Mesh.CylMesh([hx, 1., hz], '0CC')

        self.m0 = np.exp(np.random.rand(self.mesh.nC))

        src = FDEM.Src.MagDipole(
            rxList=[], loc=np.r_[0., 0., 0.], freq=self.freq
        )

        self.prob = FDEM.Problem3D_e(
            self.mesh, sigma=self.m0
        )

    def test_Aderiv(self, prbtype='e'):

        tInd = 2
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
            AderivFun, mu_0*np.random.rand(self.mesh.nC), plotIt=False,
            num=3, eps=1e-20
            )

if __name__ == '__main__':
    unittest.main()


