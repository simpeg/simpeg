import unittest
import pytest
import numpy as np

import discretize
from SimPEG import (
    maps,
    directives,
    regularization,
    data_misfit,
    optimization,
    inversion,
    inverse_problem,
    simulation,
)
from SimPEG.potential_fields import magnetics as mag
import shutil


class directivesValidation(unittest.TestCase):
    def test_validation_pass(self):
        betaest = directives.BetaEstimate_ByEig()

        IRLS = directives.Update_IRLS(f_min_change=1e-4, minGNiter=3, beta_tol=1e-2)
        update_Jacobi = directives.UpdatePreconditioner()
        dList = [betaest, IRLS, update_Jacobi]
        directiveList = directives.DirectiveList(*dList)

        self.assertTrue(directiveList.validate())

    def test_validation_fail(self):
        betaest = directives.BetaEstimate_ByEig()

        IRLS = directives.Update_IRLS(f_min_change=1e-4, minGNiter=3, beta_tol=1e-2)
        update_Jacobi = directives.UpdatePreconditioner()
        dList = [betaest, update_Jacobi, IRLS]
        directiveList = directives.DirectiveList(*dList)

        with self.assertRaises(AssertionError):
            self.assertTrue(directiveList.validate())

    def test_validation_initial_beta_fail(self):
        beta_1 = directives.BetaEstimateMaxDerivative()
        beta_2 = directives.BetaEstimate_ByEig()

        dList = [beta_1, beta_2]
        directiveList = directives.DirectiveList(*dList)
        with self.assertRaises(AssertionError):
            self.assertTrue(directiveList.validate())

    def test_validation_warning(self):
        betaest = directives.BetaEstimate_ByEig()

        IRLS = directives.Update_IRLS(f_min_change=1e-4, minGNiter=3, beta_tol=1e-2)
        dList = [betaest, IRLS]
        directiveList = directives.DirectiveList(*dList)

        with pytest.warns(UserWarning):
            self.assertTrue(directiveList.validate())


class ValidationInInversion(unittest.TestCase):
    def setUp(self):
        mesh = discretize.TensorMesh([4, 4, 4])

        # Magnetic inducing field parameter (A,I,D)
        B = [50000, 90, 0]

        # Create a MAGsurvey
        rx = mag.Point(np.vstack([[0.25, 0.25, 0.25], [-0.25, -0.25, 0.25]]))
        srcField = mag.UniformBackgroundField([rx], parameters=(B[0], B[1], B[2]))
        survey = mag.Survey(srcField)

        # Create the forward model operator
        sim = mag.Simulation3DIntegral(
            mesh, survey=survey, chiMap=maps.IdentityMap(mesh)
        )

        m = np.random.rand(mesh.nC)

        data = sim.make_synthetic_data(m, add_noise=True)
        dmis = data_misfit.L2DataMisfit(data=data, simulation=sim)
        dmis.W = 1.0 / data.relative_error

        # Add directives to the inversion
        opt = optimization.ProjectedGNCG(
            maxIter=2, lower=-10.0, upper=10.0, maxIterCG=2
        )

        self.model = m
        self.mesh = mesh
        self.dmis = dmis
        self.opt = opt
        self.sim = sim

    def test_validation_in_inversion(self):
        reg = regularization.Sparse(self.mesh)
        reg.reference_model = np.zeros(self.mesh.nC)
        reg.norms = [0, 1, 1, 1]
        reg.eps_p, reg.eps_q = 1e-3, 1e-3

        invProb = inverse_problem.BaseInvProblem(self.dmis, reg, self.opt)

        betaest = directives.BetaEstimate_ByEig()

        # Here is where the norms are applied
        IRLS = directives.Update_IRLS(f_min_change=1e-4, minGNiter=3, beta_tol=1e-2)

        update_Jacobi = directives.UpdatePreconditioner()
        sensitivity_weights = directives.UpdateSensitivityWeights()
        with self.assertRaises(AssertionError):
            # validation should happen and this will fail
            # (IRLS needs to be before update_Jacobi)
            inv = inversion.BaseInversion(
                invProb, directiveList=[betaest, update_Jacobi, IRLS]
            )

        with self.assertRaises(AssertionError):
            # validation should happen and this will fail
            # (sensitivity_weights needs to be before betaest)
            inv = inversion.BaseInversion(
                invProb, directiveList=[betaest, sensitivity_weights]
            )

        with self.assertRaises(AssertionError):
            # validation should happen and this will fail
            # (sensitivity_weights needs to be before update_Jacobi)
            inv = inversion.BaseInversion(invProb)
            inv.directiveList = [update_Jacobi, sensitivity_weights]

    def test_sensitivity_weighting_warnings(self):
        # Test setter warnings
        d_temp = directives.UpdateSensitivityWeights()
        d_temp.normalization_method = True
        self.assertTrue(d_temp.normalization_method == "maximum")

        d_temp.normalization_method = False
        self.assertTrue(d_temp.normalization_method is None)

    def test_sensitivity_weighting_global(self):
        test_inputs = {
            "everyIter": False,
            "threshold": 1e-12,
            "threshold_method": "global",
            "normalization": False,
        }

        # Compute test weights
        sqrt_diagJtJ = (
            np.sqrt(np.sum((self.dmis.W * self.sim.G) ** 2, axis=0))
            / self.mesh.cell_volumes
        )
        test_weights = sqrt_diagJtJ + test_inputs["threshold"]
        test_weights *= self.mesh.cell_volumes

        # Test directive
        reg = regularization.WeightedLeastSquares(self.mesh)
        invProb = inverse_problem.BaseInvProblem(self.dmis, reg, self.opt)
        invProb.model = self.model

        test_directive = directives.UpdateSensitivityWeights(**test_inputs)
        test_directive.inversion = inversion.BaseInversion(
            invProb, directiveList=[test_directive]
        )
        test_directive.update()

        for reg_i in reg.objfcts:
            self.assertTrue(np.all(np.isclose(test_weights, reg_i.cell_weights)))
            reg_i.remove_weights("sensitivity")

        # self.test_sensitivity_weighting_subroutine(test_weights, test_directive)

        print("GLOBAL SENSITIVITY WEIGHTING TEST PASSED")

    def test_sensitivity_weighting_percentile_maximum(self):
        test_inputs = {
            "every_iteration": True,
            "threshold_value": 1,
            "threshold_method": "percentile",
            "normalization": True,
        }

        # Compute test weights
        sqrt_diagJtJ = (
            np.sqrt(np.sum((self.dmis.W * self.sim.G) ** 2, axis=0))
            / self.mesh.cell_volumes
        )
        test_weights = np.clip(
            sqrt_diagJtJ,
            a_min=np.percentile(sqrt_diagJtJ, test_inputs["threshold_value"]),
            a_max=np.inf,
        )
        test_weights /= test_weights.max()
        test_weights *= self.mesh.cell_volumes

        # Test directive
        reg = regularization.WeightedLeastSquares(self.mesh)
        invProb = inverse_problem.BaseInvProblem(self.dmis, reg, self.opt)
        invProb.model = self.model

        test_directive = directives.UpdateSensitivityWeights(**test_inputs)
        test_directive.inversion = inversion.BaseInversion(
            invProb, directiveList=[test_directive]
        )
        test_directive.update()

        for reg_i in reg.objfcts:
            self.assertTrue(np.all(np.isclose(test_weights, reg_i.cell_weights)))
            reg_i.remove_weights("sensitivity")

        # self.test_sensitivity_weighting_subroutine(test_weights, test_directive)

        print("SENSITIVITY WEIGHTING BY PERCENTILE AND MIN VALUE TEST PASSED")

    def test_sensitivity_weighting_amplitude_minimum(self):
        test_inputs = {
            "every_iteration": True,
            "threshold_value": 1e-3,
            "threshold_method": "amplitude",
            "normalization_method": "minimum",
        }

        # Compute test weights
        sqrt_diagJtJ = (
            np.sqrt(np.sum((self.dmis.W * self.sim.G) ** 2, axis=0))
            / self.mesh.cell_volumes
        )
        test_weights = np.clip(
            sqrt_diagJtJ,
            a_min=test_inputs["threshold_value"] * sqrt_diagJtJ.max(),
            a_max=np.inf,
        )
        test_weights /= test_weights.min()
        test_weights *= self.mesh.cell_volumes

        # Test directive
        reg = regularization.WeightedLeastSquares(self.mesh)
        invProb = inverse_problem.BaseInvProblem(self.dmis, reg, self.opt)
        invProb.model = self.model

        test_directive = directives.UpdateSensitivityWeights(**test_inputs)
        test_directive.inversion = inversion.BaseInversion(
            invProb, directiveList=[test_directive]
        )
        test_directive.update()

        for reg_i in reg.objfcts:
            self.assertTrue(np.all(np.isclose(test_weights, reg_i.cell_weights)))
            reg_i.remove_weights("sensitivity")

        # self.test_sensitivity_weighting_subroutine(test_weights, test_directive)

        print("SENSITIVITY WEIGHTING BY AMPLIUTDE AND MAX ALUE TEST PASSED")

    def tearDown(self):
        # Clean up the working directory
        try:
            shutil.rmtree(self.sim.sensitivity_path)
        except FileNotFoundError:
            pass


@pytest.mark.parametrize(
    "RegClass", [regularization.Sparse, regularization.WeightedLeastSquares]
)
def test_save_output_dict(RegClass):
    mesh = discretize.TensorMesh([30])
    sim = simulation.ExponentialSinusoidSimulation(
        mesh=mesh, model_map=maps.IdentityMap()
    )
    data = sim.make_synthetic_data(np.ones(mesh.n_cells), add_noise=True)
    dmis = data_misfit.L2DataMisfit(data, sim)

    opt = optimization.InexactGaussNewton(maxIter=1)

    m_ref = np.zeros(mesh.n_cells)
    reg = RegClass(mesh, reference_model=m_ref)

    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=1)

    save_direct = directives.SaveOutputDictEveryIteration()
    inv = inversion.BaseInversion(inv_prob, directiveList=[save_direct])

    inv.run(np.zeros(mesh.n_cells))

    out_dict = save_direct.outDict[1]
    assert "iter" in out_dict
    assert "beta" in out_dict
    assert "phi_d" in out_dict
    assert "phi_m" in out_dict
    assert "f" in out_dict
    assert "m" in out_dict
    assert "dpred" in out_dict
    if RegClass is regularization.Sparse:
        assert "SparseSmallness.irls_threshold" in out_dict
        assert "SparseSmallness.norm" in out_dict
        assert "x SparseSmoothness.irls_threshold" in out_dict
        assert "x SparseSmoothness.norm" in out_dict


if __name__ == "__main__":
    unittest.main()
