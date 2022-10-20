import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.constants import mu_0
import unittest

# SimPEG, discretize
import discretize
from discretize import utils
from SimPEG.electromagnetics import time_domain as tdem
from SimPEG import maps, tests
from pymatsolver import Pardiso


class TestGroundedSourceTDEM_j(unittest.TestCase):

    prob_type = "CurrentDensity"

    @classmethod
    def setUpClass(self):

        # mesh
        cs = 10
        npad = 4
        ncore = 5
        h = [(cs, npad, -1.5), (cs, ncore), (cs, npad, 1.5)]
        mesh = discretize.TensorMesh([h, h, h], x0="CCC")

        # source
        src_a = np.r_[-cs * 2, 0.0, 0.0]
        src_b = np.r_[cs * 2, 0.0, 0.0]

        s_e = np.zeros(mesh.nFx)
        src_inds = (
            (mesh.gridFx[:, 0] >= src_a[0])
            & (mesh.gridFx[:, 0] <= src_b[0])
            & (mesh.gridFx[:, 1] >= src_a[1])
            & (mesh.gridFx[:, 1] <= src_b[1])
            & (mesh.gridFx[:, 2] >= src_a[2])
            & (mesh.gridFx[:, 2] <= src_b[2])
        )
        s_e[src_inds] = 1.0
        s_e = np.hstack([s_e, np.zeros(mesh.nFy + mesh.nFz)])

        # define a model with a conductive, permeable target
        sigma0 = 1e-1
        sigma1 = 1

        mu0 = mu_0
        mu1 = 100 * mu_0

        h_target = np.r_[-30, 30]
        target_inds = (
            (mesh.gridCC[:, 0] >= h_target[0])
            & (mesh.gridCC[:, 0] <= h_target[1])
            & (mesh.gridCC[:, 1] >= h_target[0])
            & (mesh.gridCC[:, 1] <= h_target[1])
            & (mesh.gridCC[:, 2] >= h_target[0])
            & (mesh.gridCC[:, 2] <= h_target[1])
        )

        sigma = sigma0 * np.ones(mesh.nC)
        sigma[target_inds] = sigma1

        mu = mu0 * np.ones(mesh.nC)
        mu[target_inds] = mu1

        src = tdem.Src.RawVec_Grounded([], s_e=s_e)

        time_steps = [
            (1e-6, 20),
            (1e-5, 30),
            (3e-5, 30),
            (1e-4, 40),
            (3e-4, 30),
            (1e-3, 20),
            (1e-2, 17),
        ]
        prob = getattr(tdem, "Simulation3D{}".format(self.prob_type))(
            mesh,
            time_steps=time_steps,
            mu=mu,
            sigmaMap=maps.ExpMap(mesh),
            solver=Pardiso,
        )
        survey = tdem.Survey([src])

        prob.model = sigma

        self.mesh = mesh
        self.prob = prob
        self.survey = survey
        self.src = src

        self.sigma = sigma
        self.mu = mu

        print("Testing problem {} \n\n".format(self.prob_type))

    def derivtest(self, deriv_fct):
        m0 = np.log(self.sigma) + np.random.rand(self.mesh.nC)
        self.prob.model = m0

        return tests.checkDerivative(deriv_fct, np.log(self.sigma), num=3, plotIt=False)

    def test_deriv_phi(self):
        def deriv_check(m):
            self.prob.model = m
            return [
                self.src.phiInitial(self.prob),
                lambda mx: self.src._phiInitialDeriv(self.prob, v=mx),
            ]

        self.derivtest(deriv_check)

    def test_deriv_j(self):
        def deriv_check(m):
            self.prob.model = m
            return [
                self.src.jInitial(self.prob),
                lambda mx: self.src.jInitialDeriv(self.prob, v=mx),
            ]

        self.derivtest(deriv_check)

    def test_deriv_h(self):
        def deriv_check(m):
            self.prob.model = m
            return [
                self.src.hInitial(self.prob),
                lambda mx: self.src.hInitialDeriv(self.prob, v=mx),
            ]

        self.derivtest(deriv_check)

    def test_adjoint_phi(self):

        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(self.mesh.nC)
        a = w.T.dot(self.src._phiInitialDeriv(self.prob, v=v))
        b = v.T.dot(self.src._phiInitialDeriv(self.prob, v=w, adjoint=True))
        self.assertTrue(np.allclose(a, b))

    def test_adjoint_j(self):

        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(self.mesh.nF)
        a = w.T.dot(self.src.jInitialDeriv(self.prob, v=v))
        b = v.T.dot(self.src.jInitialDeriv(self.prob, v=w, adjoint=True))
        self.assertTrue(np.allclose(a, b))

    def test_adjoint_h(self):
        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(self.mesh.nE)
        a = w.T.dot(self.src.hInitialDeriv(self.prob, v=v))
        b = v.T.dot(self.src.hInitialDeriv(self.prob, v=w, adjoint=True))
        self.assertTrue(np.allclose(a, b))


class TestGroundedSourceTDEM_h(TestGroundedSourceTDEM_j):

    prob_type = "MagneticField"
