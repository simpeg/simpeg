import pytest
import unittest
import numpy as np
from scipy.sparse.linalg import eigsh
from discretize import TensorMesh
from simpeg.objective_function import BaseObjectiveFunction
from simpeg import simulation, data_misfit
from simpeg.maps import IdentityMap
from simpeg.regularization import WeightedLeastSquares
from simpeg.directives.directives import BetaEstimate_ByEig


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
            return np.exp(p * jk[k] * mesh.cell_centers_x) * np.cos(
                np.pi * q * jk[k] * mesh.cell_centers_x
            )

        G = np.empty((nk, mesh.nC))

        for i in range(nk):
            G[i, :] = g(i)
        self.G = G

        # Creating the true model
        true_model = np.zeros(mesh.nC)
        true_model[mesh.cell_centers_x > 0.3] = 1.0
        true_model[mesh.cell_centers_x > 0.45] = -0.5
        true_model[mesh.cell_centers_x > 0.6] = 0
        self.true_model = true_model

        # Create a simpeg simulation
        model_map = IdentityMap(mesh)
        sim = simulation.LinearSimulation(mesh, G=G, model_map=model_map)

        # Create a simpeg data object
        relative_error = 0.1
        noise_floor = 1e-4
        data_obj = sim.make_synthetic_data(
            true_model,
            relative_error=relative_error,
            noise_floor=noise_floor,
            add_noise=True,
            random_seed=40,
        )
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=data_obj)
        self.dmis = dmis

        # Test for joint misfits
        n_misfits = 5
        multipliers = np.random.randn(n_misfits) ** 2
        multipliers /= np.sum(multipliers)
        self.multipliers = multipliers
        dmiscombo = dmis
        for mult in multipliers:
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
        dmis_matrix = 2 * self.G.T.dot((self.dmis.W**2).dot(self.G))
        field = self.dmis.simulation.fields(self.true_model)
        max_eigenvalue_numpy, _ = eigsh(dmis_matrix, k=1)
        max_eigenvalue_directive = BetaEstimate_ByEig.eigenvalue_by_power_iteration(
            self.dmis, self.true_model, fields_list=field, n_pw_iter=30, random_seed=42
        )
        passed = np.isclose(max_eigenvalue_numpy, max_eigenvalue_directive, rtol=1e-2)
        self.assertTrue(passed, True)
        print("Eigenvalue Utils for one data misfit term is validated.")

        # Test for multiple data misfit
        WtW = 0.0
        for mult, dm in zip(self.dmiscombo.multipliers, self.dmiscombo.objfcts):
            WtW += mult * dm.W**2
        dmiscombo_matrix = 2 * self.G.T.dot(WtW.dot(self.G))
        max_eigenvalue_numpy, _ = eigsh(dmiscombo_matrix, k=1)
        max_eigenvalue_directive = BetaEstimate_ByEig.eigenvalue_by_power_iteration(
            self.dmiscombo, self.true_model, n_pw_iter=30, random_seed=42
        )
        passed = np.isclose(max_eigenvalue_numpy, max_eigenvalue_directive, rtol=1e-2)
        self.assertTrue(passed, True)
        print("Eigenvalue Utils for multiple data misfit terms is validated.")

    def test_reg_eigenvalue_by_power_iteration(self):
        reg_maxtrix = self.reg.deriv2(self.true_model)
        max_eigenvalue_numpy, _ = eigsh(reg_maxtrix, k=1)
        max_eigenvalue_directive = BetaEstimate_ByEig.eigenvalue_by_power_iteration(
            self.reg, self.true_model, n_pw_iter=100, random_seed=42
        )
        passed = np.isclose(max_eigenvalue_numpy, max_eigenvalue_directive, rtol=1e-2)
        self.assertTrue(passed, True)
        print("Eigenvalue Utils for regularization is validated.")

    def test_combo_eigenvalue_by_power_iteration(self):
        reg_maxtrix = self.reg.deriv2(self.true_model)
        dmis_matrix = 2 * self.G.T.dot((self.dmis.W**2).dot(self.G))
        combo_matrix = dmis_matrix + self.beta * reg_maxtrix
        max_eigenvalue_numpy, _ = eigsh(combo_matrix, k=1)
        max_eigenvalue_directive = BetaEstimate_ByEig.eigenvalue_by_power_iteration(
            self.mixcombo, self.true_model, n_pw_iter=100, random_seed=42
        )
        passed = np.isclose(max_eigenvalue_numpy, max_eigenvalue_directive, rtol=1e-2)
        self.assertTrue(passed, True)
        print("Eigenvalue Utils for a mixed ComboObjectiveFunction is validated.")


class TestDeprecatedSeed:
    """Test deprecation of ``seed`` argument."""

    @pytest.fixture
    def mock_objfun(self):
        """
        Mock objective function class as child of ``BaseObjectiveFunction``
        """

        class MockObjectiveFunction(BaseObjectiveFunction):

            def deriv2(self, m, v=None, **kwargs):
                return np.ones(self.nP)

        return MockObjectiveFunction

    def get_message_duplicated_error(self, old_name, new_name, version="v0.24.0"):
        msg = (
            f"Cannot pass both '{new_name}' and '{old_name}'."
            f"'{old_name}' has been deprecated and will be removed in "
            f" SimPEG {version}, please use '{new_name}' instead."
        )
        return msg

    def get_message_deprecated_warning(self, old_name, new_name, version="v0.24.0"):
        msg = (
            f"'{old_name}' has been deprecated and will be removed in "
            f" SimPEG {version}, please use '{new_name}' instead."
        )
        return msg

    def test_warning_argument(self, mock_objfun):
        """
        Test if warning is raised after passing ``seed``.
        """
        msg = self.get_message_deprecated_warning("seed", "random_seed")
        n_params = 5
        combo = mock_objfun(nP=n_params) + 3.0 * mock_objfun(nP=n_params)
        model = np.ones(n_params)
        with pytest.warns(FutureWarning, match=msg):
            result_seed = BetaEstimate_ByEig.eigenvalue_by_power_iteration(
                combo_objfct=combo, model=model, seed=42
            )
        # Ensure that using `seed` and `random_seed` generate the same output
        result_random_seed = BetaEstimate_ByEig.eigenvalue_by_power_iteration(
            combo_objfct=combo, model=model, random_seed=42
        )
        np.testing.assert_allclose(result_seed, result_random_seed)

    def test_error_duplicated_argument(self):
        """
        Test error after passing ``seed`` and ``random_seed``.
        """
        msg = self.get_message_duplicated_error("seed", "random_seed")
        with pytest.raises(TypeError, match=msg):
            BetaEstimate_ByEig.eigenvalue_by_power_iteration(
                combo_objfct=None, model=None, random_seed=42, seed=42
            )


if __name__ == "__main__":
    unittest.main()
