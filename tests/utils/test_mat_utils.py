from __future__ import print_function

import unittest
import numpy as np
from scipy.sparse.linalg import eigsh
from discretize import TensorMesh
from SimPEG import simulation, data_misfit
from SimPEG.maps import IdentityMap
from SimPEG.regularization import WeightedLeastSquares
from SimPEG.utils.mat_utils import eigenvalue_by_power_iteration


class TestEigenvalues(unittest.TestCase):
    def setUp(self):
        # Mesh
        N = 100
        mesh = TensorMesh([N])

        # Survey design parameters
        nk = 30
        jk = np.linspace(1.0, 59.0, nk)
        p = -0.25
        q = 0.25

        # Physics
        def g(k):
            return np.exp(p * jk[k] * mesh.vectorCCx) * np.cos(
                np.pi * q * jk[k] * mesh.vectorCCx
            )

        G = np.empty((nk, mesh.nC))

        for i in range(nk):
            G[i, :] = g(i)
        self.G = G

        # Creating the true model
        true_model = np.zeros(mesh.nC)
        true_model[mesh.vectorCCx > 0.3] = 1.0
        true_model[mesh.vectorCCx > 0.45] = -0.5
        true_model[mesh.vectorCCx > 0.6] = 0
        self.true_model = true_model

        # Create a SimPEG simulation
        model_map = IdentityMap(mesh)
        sim = simulation.LinearSimulation(mesh, G=G, model_map=model_map)

        # Create a SimPEG data object
        relative_error = 0.1
        noise_floor = 1e-4
        data_obj = sim.make_synthetic_data(
            true_model,
            relative_error=relative_error,
            noise_floor=noise_floor,
            add_noise=True,
        )
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=data_obj)
        self.dmis = dmis

        # Test for joint misfits
        n_misfits = 5
        multipliers = np.random.randn(n_misfits) ** 2
        multipliers /= np.sum(multipliers)
        self.multipliers = multipliers
        dmiscombo = dmis
        for i, mult in enumerate(multipliers):
            dmiscombo += mult * dmis
        self.dmiscombo = dmiscombo

        # Test for a regularization term
        reg = WeightedLeastSquares(mesh=mesh)
        self.reg = reg

        # Test a mix combo
        self.beta = 10.0
        self.mixcombo = self.dmis + self.beta * self.reg

    def test_dm_eigenvalue_by_power_iteration(self):
        # Test for a single data misfit
        dmis_matrix = self.G.T.dot((self.dmis.W ** 2).dot(self.G))
        field = self.dmis.simulation.fields(self.true_model)
        max_eigenvalue_numpy, _ = eigsh(dmis_matrix, k=1)
        max_eigenvalue_directive = eigenvalue_by_power_iteration(
            self.dmis, self.true_model, fields_list=field, n_pw_iter=30
        )
        passed = np.isclose(max_eigenvalue_numpy, max_eigenvalue_directive, rtol=1e-2)
        self.assertTrue(passed, True)
        print("Eigenvalue Utils for one data misfit term is validated.")

        # Test for multiple data misfit
        WtW = 0.0
        for i, (mult, dm) in enumerate(
            zip(self.dmiscombo.multipliers, self.dmiscombo.objfcts)
        ):
            WtW += mult * dm.W ** 2
        dmiscombo_matrix = self.G.T.dot(WtW.dot(self.G))
        max_eigenvalue_numpy, _ = eigsh(dmiscombo_matrix, k=1)
        max_eigenvalue_directive = eigenvalue_by_power_iteration(
            self.dmiscombo, self.true_model, n_pw_iter=30
        )
        passed = np.isclose(max_eigenvalue_numpy, max_eigenvalue_directive, rtol=1e-2)
        self.assertTrue(passed, True)
        print("Eigenvalue Utils for multiple data misfit terms is validated.")

    def test_reg_eigenvalue_by_power_iteration(self):
        reg_maxtrix = self.reg.deriv2(self.true_model)
        max_eigenvalue_numpy, _ = eigsh(reg_maxtrix, k=1)
        max_eigenvalue_directive = eigenvalue_by_power_iteration(
            self.reg, self.true_model, n_pw_iter=100
        )
        passed = np.isclose(max_eigenvalue_numpy, max_eigenvalue_directive, rtol=1e-2)
        self.assertTrue(passed, True)
        print("Eigenvalue Utils for regularization is validated.")

    def test_combo_eigenvalue_by_power_iteration(self):
        reg_maxtrix = self.reg.deriv2(self.true_model)
        dmis_matrix = self.G.T.dot((self.dmis.W ** 2).dot(self.G))
        combo_matrix = dmis_matrix + self.beta * reg_maxtrix
        max_eigenvalue_numpy, _ = eigsh(combo_matrix, k=1)
        max_eigenvalue_directive = eigenvalue_by_power_iteration(
            self.mixcombo, self.true_model, n_pw_iter=100
        )
        passed = np.isclose(max_eigenvalue_numpy, max_eigenvalue_directive, rtol=1e-2)
        self.assertTrue(passed, True)
        print("Eigenvalue Utils for a mixed ComboObjectiveFunction is validated.")


if __name__ == "__main__":
    unittest.main()
