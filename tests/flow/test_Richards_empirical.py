from __future__ import print_function
import unittest
import numpy as np
import properties
from SimPEG import Mesh
from SimPEG import Maps
from SimPEG.Tests import checkDerivative
from SimPEG.FLOW import Richards


TOL = 1E-8

np.random.seed(2)


class TestModels(unittest.TestCase):

    def test_haverkamp_theta(self):
        mesh = Mesh.TensorMesh([50])
        hav = Richards.Empirical.Haverkamp_theta(mesh)
        passed = checkDerivative(
            lambda u: (hav(u), hav.derivU(u)),
            np.random.randn(50),
            plotIt=False
        )
        self.assertTrue(passed, True)

    def test_vangenuchten_theta(self):
        mesh = Mesh.TensorMesh([50])
        van = Richards.Empirical.Vangenuchten_theta(mesh)
        passed = checkDerivative(
            lambda u: (van(u), van.derivU(u)),
            np.random.randn(50),
            plotIt=False
        )
        self.assertTrue(passed, True)

    def test_haverkamp_k(self):

        mesh = Mesh.TensorMesh([5])
        wires2 = Maps.Wires(('one', mesh.nC), ('two', mesh.nC))
        wires3 = Maps.Wires(
            ('one', mesh.nC), ('two', mesh.nC), ('three', mesh.nC)
        )
        expmap = Maps.IdentityMap(nP=mesh.nC)

        hav = Richards.Empirical.Haverkamp_k(mesh)

        print('Haverkamp_k test u deriv')
        passed = checkDerivative(
            lambda u: (hav(u), hav.derivU(u)),
            np.random.randn(mesh.nC),
            plotIt=False
        )
        self.assertTrue(passed, True)

        opts = [
            ('Ks',
                dict(KsMap=expmap), 1),
            ('A',
                dict(AMap=expmap), 1),
            ('gamma',
                dict(gammaMap=expmap), 1),
            ('Ks-A',
                dict(KsMap=expmap*wires2.one, AMap=expmap*wires2.two), 2),
            ('Ks-gamma',
                dict(KsMap=expmap*wires2.one, gammaMap=expmap*wires2.two), 2),
            ('A-gamma',
                dict(AMap=expmap*wires2.one, gammaMap=expmap*wires2.two), 2),
            ('Ks-A-gamma', dict(
                KsMap=expmap*wires3.one,
                AMap=expmap*wires3.two,
                gammaMap=expmap*wires3.three), 3),
        ]

        u = np.random.randn(mesh.nC)

        for name, opt, nM in opts:
            np.random.seed(2)
            hav = Richards.Empirical.Haverkamp_k(mesh, **opt)

            def fun(m):
                hav.model = m
                return hav(u), hav.derivM(u)

            print('Haverkamp_k test m deriv:  ', name)

            passed = checkDerivative(
                fun,
                np.random.randn(mesh.nC * nM),
                plotIt=False
            )
            self.assertTrue(passed, True)

    def test_vangenuchten_k(self):
        mesh = Mesh.TensorMesh([50])
        expmap = Maps.ExpMap(nP=50)
        van = Richards.Empirical.Vangenuchten_k(mesh, KsMap=expmap)

        m = np.random.randn(50)
        van.model = m
        passed = checkDerivative(
            lambda u: (van(u), van.derivU(u)),
            np.random.randn(50),
            plotIt=False
        )
        self.assertTrue(passed, True)

        hav = Richards.Empirical.Vangenuchten_k(mesh, KsMap=expmap)

        u = np.random.randn(mesh.nC)

        def fun(m):
            hav.model = m
            return hav(u), hav.derivM(u)

        passed = checkDerivative(
            fun,
            np.random.randn(50),
            plotIt=False
        )

        self.assertTrue(passed, True)

if __name__ == '__main__':
    unittest.main()
