from __future__ import print_function
import unittest
import numpy as np
from SimPEG import Mesh
from SimPEG import Maps
from SimPEG.Tests import checkDerivative
from SimPEG.FLOW import Richards


TOL = 1E-8

np.random.seed(2)


class TestModels(unittest.TestCase):

    def test_haverkamp_theta_u(self):
        mesh = Mesh.TensorMesh([50])
        hav = Richards.Empirical.Haverkamp_theta(mesh)
        passed = checkDerivative(
            lambda u: (hav(u), hav.derivU(u)),
            np.random.randn(50),
            plotIt=False
        )
        self.assertTrue(passed, True)

    def test_haverkamp_theta_m(self):
        mesh = Mesh.TensorMesh([50])
        idnmap = Maps.IdentityMap(nP=mesh.nC)

        seeds = {
            'theta_r': np.random.rand(mesh.nC),
            'theta_s': np.random.rand(mesh.nC),
            'alpha': np.random.rand(mesh.nC),
            'beta': np.random.rand(mesh.nC),
        }

        opts = [
            ('theta_r',
                dict(theta_rMap=idnmap), 1),
            ('theta_s',
                dict(theta_sMap=idnmap), 1),
            ('alpha',
                dict(alphaMap=idnmap), 1),
            ('beta',
                dict(betaMap=idnmap), 1),
        ]

        u = np.random.randn(mesh.nC)

        for name, opt, nM in opts:
            van = Richards.Empirical.Haverkamp_theta(mesh, **opt)

            x0 = np.concatenate([seeds[n] for n in name.split('-')])

            def fun(m):
                van.model = m
                return van(u), van.derivM(u)

            print('Haverkamp_theta test m deriv:  ', name)

            passed = checkDerivative(
                fun,
                x0,
                plotIt=False
            )
            self.assertTrue(passed, True)

    def test_vangenuchten_theta_u(self):
        mesh = Mesh.TensorMesh([50])
        van = Richards.Empirical.Vangenuchten_theta(mesh)
        passed = checkDerivative(
            lambda u: (van(u), van.derivU(u)),
            np.random.randn(50),
            plotIt=False
        )
        self.assertTrue(passed, True)

    def test_vangenuchten_theta_m(self):
        mesh = Mesh.TensorMesh([50])
        idnmap = Maps.IdentityMap(nP=mesh.nC)

        seeds = {
            'theta_r': np.random.rand(mesh.nC),
            'theta_s': np.random.rand(mesh.nC),
            'n': np.random.rand(mesh.nC) + 1,
            'alpha': np.random.rand(mesh.nC),
        }

        opts = [
            ('theta_r',
                dict(theta_rMap=idnmap), 1),
            ('theta_s',
                dict(theta_sMap=idnmap), 1),
            ('n',
                dict(nMap=idnmap), 1),
            ('alpha',
                dict(alphaMap=idnmap), 1),
        ]

        u = np.random.randn(mesh.nC)

        for name, opt, nM in opts:
            van = Richards.Empirical.Vangenuchten_theta(mesh, **opt)

            x0 = np.concatenate([seeds[n] for n in name.split('-')])

            def fun(m):
                van.model = m
                return van(u), van.derivM(u)

            print('Vangenuchten_theta test m deriv:  ', name)

            passed = checkDerivative(
                fun,
                x0,
                plotIt=False
            )
            self.assertTrue(passed, True)

    def test_haverkamp_k_u(self):

        mesh = Mesh.TensorMesh([5])

        hav = Richards.Empirical.Haverkamp_k(mesh)
        print('Haverkamp_k test u deriv')
        passed = checkDerivative(
            lambda u: (hav(u), hav.derivU(u)),
            np.random.randn(mesh.nC),
            plotIt=False
        )
        self.assertTrue(passed, True)

    def test_haverkamp_k_m(self):

        mesh = Mesh.TensorMesh([5])
        expmap = Maps.IdentityMap(nP=mesh.nC)
        wires2 = Maps.Wires(('one', mesh.nC), ('two', mesh.nC))
        wires3 = Maps.Wires(
            ('one', mesh.nC), ('two', mesh.nC), ('three', mesh.nC)
        )

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

    def test_vangenuchten_k_u(self):
        mesh = Mesh.TensorMesh([50])

        van = Richards.Empirical.Vangenuchten_k(mesh)

        print('Vangenuchten_k test u deriv')
        passed = checkDerivative(
            lambda u: (van(u), van.derivU(u)),
            np.random.randn(mesh.nC),
            plotIt=False
        )
        self.assertTrue(passed, True)

    def test_vangenuchten_k_m(self):
        mesh = Mesh.TensorMesh([50])

        expmap = Maps.ExpMap(nP=mesh.nC)
        idnmap = Maps.IdentityMap(nP=mesh.nC)

        seeds = {
            'Ks': np.random.triangular(
                np.log(1e-7), np.log(1e-6), np.log(1e-5), mesh.nC
            ),
            'I': np.random.rand(mesh.nC),
            'n': np.random.rand(mesh.nC) + 1,
            'alpha': np.random.rand(mesh.nC),
        }

        opts = [
            ('Ks',
                dict(KsMap=expmap), 1),
            ('I',
                dict(IMap=idnmap), 1),
            ('n',
                dict(nMap=idnmap), 1),
            ('alpha',
                dict(alphaMap=idnmap), 1),
        ]

        u = np.random.randn(mesh.nC)

        for name, opt, nM in opts:
            van = Richards.Empirical.Vangenuchten_k(mesh, **opt)

            x0 = np.concatenate([seeds[n] for n in name.split('-')])

            def fun(m):
                van.model = m
                return van(u), van.derivM(u)

            print('Vangenuchten_k test m deriv:  ', name)

            passed = checkDerivative(
                fun,
                x0,
                plotIt=False
            )
            self.assertTrue(passed, True)

if __name__ == '__main__':
    unittest.main()
