import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.constants import mu_0
import unittest

# SimPEG, discretize
import discretize
from discretize import utils
from SimPEG.EM import TDEM
from SimPEG import Utils, Maps, Tests
from pymatsolver import Pardiso


class TestGroundedSourceTDEM(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # mesh
        cs = 10
        npad = 5
        ncore = 5
        h = [(cs, npad, -1.5), (cs, ncore), (cs, npad, 1.5)]
        mesh = discretize.TensorMesh([h, h, h], x0="CCC")

        # source
        src_a = np.r_[-cs*2, 0., 0.]
        src_b = np.r_[cs*2, 0., 0.]

        s_e = np.zeros(mesh.nFx)
        src_inds = (
            (mesh.gridFx[:, 0] >= src_a[0]) & (mesh.gridFx[:, 0] <= src_b[0]) &
            (mesh.gridFx[:, 1] >= src_a[1]) & (mesh.gridFx[:, 1] <= src_b[1]) &
            (mesh.gridFx[:, 2] >= src_a[2]) & (mesh.gridFx[:, 2] <= src_b[2])
        )
        s_e[src_inds] = 1.
        s_e = np.hstack([s_e, np.zeros(mesh.nFy + mesh.nFz)])

        # define a model with a conductive, permeable target
        sigma0 = 1e-1
        sigma1 = 1

        mu0 = mu_0
        mu1 = 100*mu_0

        h_target = np.r_[-30, 30]
        target_inds = (
            (mesh.gridCC[:, 0] >= h_target[0]) & (mesh.gridCC[:, 0] <= h_target[1]) &
            (mesh.gridCC[:, 1] >= h_target[0]) & (mesh.gridCC[:, 1] <= h_target[1]) &
            (mesh.gridCC[:, 2] >= h_target[0]) & (mesh.gridCC[:, 2] <= h_target[1])
        )

        sigma = sigma0 * np.ones(mesh.nC)
        sigma[target_inds] = sigma1

        mu = mu0 * np.ones(mesh.nC)
        mu[target_inds] = mu1

        src = TDEM.Src.RawVec_Grounded([], s_e=s_e)

        timeSteps = [
            (1e-6, 20), (1e-5, 30), (3e-5, 30), (1e-4, 40), (3e-4, 30),
            (1e-3, 20), (1e-2, 17)
        ]
        prob = TDEM.Problem3D_j(mesh, timeSteps=timeSteps, mu=mu, sigmaMap=Maps.ExpMap(mesh))
        survey = TDEM.Survey([src])

        self.mesh = mesh
        self.prob = prob
        self.survey = survey
        self.src = src

        self.sigma = sigma
        self.mu = mu

    def deriv_test(self, deriv_fct):
        m0 = np.log(self.sigma) + np.random.rand(self.mesh.nC)
        self.prob.model = m0

        Tests.checkDerivative(deriv_fct, np.log(self.sigma), num=4, plotIt=False)

    def test_deriv_phi(self):

        def deriv_check(m):
            self.prob.model = m
            return [
                self.src.phiInitial(self.prob),
                lambda mx: self.src._phiInitialDeriv(self.prob, v=mx)
            ]
        self.deriv_test(deriv_check)

    def test_deriv_j(self):

        def deriv_check(m):
            self.prob.model = m
            return [
                self.src.jInitial(self.prob),
                lambda mx: self.src.jInitialDeriv(self.prob, v=mx)
            ]
        self.deriv_test(deriv_check)

    def test_deriv_h(self):

        def deriv_check(m):
            self.prob.model = m
            return [
                self.src.hInitial(self.prob),
                lambda mx: self.src.hInitialDeriv(self.prob, v=mx)
            ]
        self.deriv_test(deriv_check)



if __name__ == '__main__':
    unittest.main()
