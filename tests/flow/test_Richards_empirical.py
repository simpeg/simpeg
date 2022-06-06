from __future__ import print_function
import unittest

import numpy as np

import discretize
from discretize.tests import check_derivative

from SimPEG import maps
from SimPEG.flow import richards

TOL = 1e-8

np.random.seed(2)


class TestModels(unittest.TestCase):
    def test_haverkamp_theta_u(self):
        mesh = discretize.TensorMesh([50])
        hav = richards.empirical.Haverkamp_theta(mesh)
        passed = check_derivative(
            lambda u: (hav(u), hav.derivU(u)), np.random.randn(50), plotIt=False
        )
        self.assertTrue(passed, True)

    def test_haverkamp_theta_m(self):
        mesh = discretize.TensorMesh([50])
        idnmap = maps.IdentityMap(nP=mesh.nC)

        seeds = {
            "theta_r": np.random.rand(mesh.nC),
            "theta_s": np.random.rand(mesh.nC),
            "alpha": np.random.rand(mesh.nC),
            "beta": np.random.rand(mesh.nC),
        }

        opts = [
            ("theta_r", dict(theta_rMap=idnmap), 1),
            ("theta_s", dict(theta_sMap=idnmap), 1),
            ("alpha", dict(alphaMap=idnmap), 1),
            ("beta", dict(betaMap=idnmap), 1),
        ]

        u = np.random.randn(mesh.nC)

        for name, opt, nM in opts:
            van = richards.empirical.Haverkamp_theta(mesh, **opt)

            x0 = np.concatenate([seeds[n] for n in name.split("-")])

            def fun(m):
                van.model = m
                return van(u), van.derivM(u)

            print("Haverkamp_theta test m deriv:  ", name)

            passed = check_derivative(fun, x0, plotIt=False)
            self.assertTrue(passed, True)

    def test_vangenuchten_theta_u(self):
        mesh = discretize.TensorMesh([50])
        van = richards.empirical.Vangenuchten_theta(mesh)
        passed = check_derivative(
            lambda u: (van(u), van.derivU(u)), np.random.randn(50), plotIt=False
        )
        self.assertTrue(passed, True)

    def test_vangenuchten_theta_m(self):
        mesh = discretize.TensorMesh([50])
        idnmap = maps.IdentityMap(nP=mesh.nC)

        seeds = {
            "theta_r": np.random.rand(mesh.nC),
            "theta_s": np.random.rand(mesh.nC),
            "n": np.random.rand(mesh.nC) + 1,
            "alpha": np.random.rand(mesh.nC),
        }

        opts = [
            ("theta_r", dict(theta_rMap=idnmap), 1),
            ("theta_s", dict(theta_sMap=idnmap), 1),
            ("n", dict(nMap=idnmap), 1),
            ("alpha", dict(alphaMap=idnmap), 1),
        ]

        u = np.random.randn(mesh.nC)

        for name, opt, nM in opts:
            van = richards.empirical.Vangenuchten_theta(mesh, **opt)

            x0 = np.concatenate([seeds[n] for n in name.split("-")])

            def fun(m):
                van.model = m
                return van(u), van.derivM(u)

            print("Vangenuchten_theta test m deriv:  ", name)

            passed = check_derivative(fun, x0, plotIt=False)
            self.assertTrue(passed, True)

    def test_haverkamp_k_u(self):

        mesh = discretize.TensorMesh([5])

        hav = richards.empirical.Haverkamp_k(mesh)
        print("Haverkamp_k test u deriv")
        passed = check_derivative(
            lambda u: (hav(u), hav.derivU(u)), np.random.randn(mesh.nC), plotIt=False
        )
        self.assertTrue(passed, True)

    def test_haverkamp_k_m(self):

        mesh = discretize.TensorMesh([5])
        expmap = maps.IdentityMap(nP=mesh.nC)
        wires2 = maps.Wires(("one", mesh.nC), ("two", mesh.nC))
        wires3 = maps.Wires(("one", mesh.nC), ("two", mesh.nC), ("three", mesh.nC))

        opts = [
            ("Ks", dict(KsMap=expmap), 1),
            ("A", dict(AMap=expmap), 1),
            ("gamma", dict(gammaMap=expmap), 1),
            ("Ks-A", dict(KsMap=expmap * wires2.one, AMap=expmap * wires2.two), 2),
            (
                "Ks-gamma",
                dict(KsMap=expmap * wires2.one, gammaMap=expmap * wires2.two),
                2,
            ),
            (
                "A-gamma",
                dict(AMap=expmap * wires2.one, gammaMap=expmap * wires2.two),
                2,
            ),
            (
                "Ks-A-gamma",
                dict(
                    KsMap=expmap * wires3.one,
                    AMap=expmap * wires3.two,
                    gammaMap=expmap * wires3.three,
                ),
                3,
            ),
        ]

        u = np.random.randn(mesh.nC)

        for name, opt, nM in opts:
            np.random.seed(2)
            hav = richards.empirical.Haverkamp_k(mesh, **opt)

            def fun(m):
                hav.model = m
                return hav(u), hav.derivM(u)

            print("Haverkamp_k test m deriv:  ", name)

            passed = check_derivative(fun, np.random.randn(mesh.nC * nM), plotIt=False)
            self.assertTrue(passed, True)

    def test_vangenuchten_k_u(self):
        mesh = discretize.TensorMesh([50])

        van = richards.empirical.Vangenuchten_k(mesh)

        print("Vangenuchten_k test u deriv")
        passed = check_derivative(
            lambda u: (van(u), van.derivU(u)), np.random.randn(mesh.nC), plotIt=False
        )
        self.assertTrue(passed, True)

    def test_vangenuchten_k_m(self):
        mesh = discretize.TensorMesh([50])

        expmap = maps.ExpMap(nP=mesh.nC)
        idnmap = maps.IdentityMap(nP=mesh.nC)

        seeds = {
            "Ks": np.random.triangular(
                np.log(1e-7), np.log(1e-6), np.log(1e-5), mesh.nC
            ),
            "I": np.random.rand(mesh.nC),
            "n": np.random.rand(mesh.nC) + 1,
            "alpha": np.random.rand(mesh.nC),
        }

        opts = [
            ("Ks", dict(KsMap=expmap), 1),
            ("I", dict(IMap=idnmap), 1),
            ("n", dict(nMap=idnmap), 1),
            ("alpha", dict(alphaMap=idnmap), 1),
        ]

        u = np.random.randn(mesh.nC)

        for name, opt, nM in opts:
            van = richards.empirical.Vangenuchten_k(mesh, **opt)

            x0 = np.concatenate([seeds[n] for n in name.split("-")])

            def fun(m):
                van.model = m
                return van(u), van.derivM(u)

            print("Vangenuchten_k test m deriv:  ", name)

            passed = check_derivative(fun, x0, plotIt=False)
            self.assertTrue(passed, True)


if __name__ == "__main__":
    unittest.main()
