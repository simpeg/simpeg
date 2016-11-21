from __future__ import print_function
import unittest
import numpy as np
from SimPEG import Mesh
from SimPEG import Utils
from SimPEG import Maps
from SimPEG.Tests import checkDerivative
from SimPEG.FLOW import Richards
try:
    from pymatsolver import PardisoSolver as Solver
except Exception:
    from SimPEG import Solver


TOL = 1E-8

np.random.seed(0)


class TestModels(unittest.TestCase):

    def test_haverkamp_theta(self):
        mesh = Mesh.TensorMesh([50])
        hav = Richards.Empirical.Haverkamp_theta(mesh)
        m = np.random.randn(50)
        passed = checkDerivative(
            lambda u: (hav(u, m), hav.derivU(u, m)),
            np.random.randn(50),
            plotIt=False
        )
        self.assertTrue(passed, True)

    def test_vangenuchten_theta(self):
        mesh = Mesh.TensorMesh([50])
        van = Richards.Empirical.Vangenuchten_theta(mesh)
        m = np.random.randn(50)
        passed = checkDerivative(
            lambda u: (van(u, m), van.derivU(u, m)),
            np.random.randn(50),
            plotIt=False
        )
        self.assertTrue(passed, True)

    def test_haverkamp_k(self):
        mesh = Mesh.TensorMesh([50])
        hav = Richards.Empirical.Haverkamp_k(mesh)
        m = np.random.randn(50)
        passed = checkDerivative(
            lambda u: (hav(u, m), hav.derivU(u, m)),
            np.random.randn(50),
            plotIt=False
        )
        self.assertTrue(passed, True)

        hav = Richards.Empirical.Haverkamp_k(mesh)
        u = np.random.randn(50)

        passed = checkDerivative(
            lambda m: (hav(u, m), hav.derivM(u, m)),
            np.random.randn(50),
            plotIt=False
        )
        self.assertTrue(passed, True)

    def test_vangenuchten_k(self):
        mesh = Mesh.TensorMesh([5])
        expmap = Maps.ExpMap(nP=5)
        van = Richards.Empirical.Vangenuchten_k(mesh, KsMap=expmap)

        m = np.random.randn(5)
        van.model = m
        print(van.KsDeriv)
        passed = checkDerivative(
            lambda u: (van(u, m), van.derivU(u, m)),
            np.random.randn(5),
            plotIt=False
        )
        self.assertTrue(passed, True)

        hav = Richards.Empirical.Vangenuchten_k(mesh, KsMap=expmap)
        u = np.random.randn(5)
        print(u)
        passed = checkDerivative(
            lambda m: (hav(u, m), hav.derivM(u, m)),
            np.random.randn(5),
            plotIt=False
        )
        self.assertTrue(passed, True)

if __name__ == '__main__':
    unittest.main()
