import unittest
import pytest
import numpy as np

import discretize
from simpeg import (
    maps,
    directives,
    regularization,
    data_misfit,
    optimization,
    inversion,
    inverse_problem,
    simulation,
)
from simpeg.potential_fields import magnetics as mag
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
        h0_amplitude, h0_inclination, h0_declination = (50000, 90, 0)

        # Create a MAGsurvey
        rx = mag.Point(np.vstack([[0.25, 0.25, 0.25], [-0.25, -0.25, 0.25]]))
        srcField = mag.UniformBackgroundField(
            receiver_list=[rx],
            amplitude=h0_amplitude,
            inclination=h0_inclination,
            declination=h0_declination,
        )
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
            inversion.BaseInversion(
                invProb, directiveList=[betaest, update_Jacobi, IRLS]
            )

        with self.assertRaises(AssertionError):
            # validation should happen and this will fail
            # (sensitivity_weights needs to be before betaest)
            inversion.BaseInversion(
                invProb, directiveList=[betaest, sensitivity_weights]
            )

        with self.assertRaises(AssertionError):
            # validation should happen and this will fail
            # (sensitivity_weights needs to be before update_Jacobi)
            inv = inversion.BaseInversion(invProb)
            inv.directiveList = [update_Jacobi, sensitivity_weights]

    def test_sensitivity_weighting_global(self):
        test_inputs = {
            "every_iteration": False,
            "threshold_value": 1e-12,
            "threshold_method": "global",
            "normalization_method": None,
        }

        # Compute test weights
        sqrt_diagJtJ = (
            np.sqrt(np.sum((self.dmis.W * self.sim.G) ** 2, axis=0))
            / self.mesh.cell_volumes
        )
        test_weights = sqrt_diagJtJ + test_inputs["threshold_value"]
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
            # Get all weights in regularization
            weights = [reg_i.get_weights(key) for key in reg_i.weights_keys]
            # Compute the product of all weights
            weights = np.prod(weights, axis=0)
            self.assertTrue(np.all(np.isclose(test_weights, weights)))
            reg_i.remove_weights("sensitivity")

        # self.test_sensitivity_weighting_subroutine(test_weights, test_directive)

        print("GLOBAL SENSITIVITY WEIGHTING TEST PASSED")

    def test_sensitivity_weighting_percentile_maximum(self):
        test_inputs = {
            "every_iteration": True,
            "threshold_value": 1,
            "threshold_method": "percentile",
            "normalization_method": "maximum",
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
            # Get all weights in regularization
            weights = [reg_i.get_weights(key) for key in reg_i.weights_keys]
            # Compute the product of all weights
            weights = np.prod(weights, axis=0)
            self.assertTrue(np.all(np.isclose(test_weights, weights)))
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
            # Get all weights in regularization
            weights = [reg_i.get_weights(key) for key in reg_i.weights_keys]
            # Compute the product of all weights
            weights = np.prod(weights, axis=0)
            self.assertTrue(np.all(np.isclose(test_weights, weights)))
            reg_i.remove_weights("sensitivity")

        # self.test_sensitivity_weighting_subroutine(test_weights, test_directive)

        print("SENSITIVITY WEIGHTING BY AMPLITUDE AND MAX VALUE TEST PASSED")

    def test_irls_directive(self):
        input_norms = [0.0, 1.0, 1.0, 1.0]
        reg = regularization.Sparse(self.mesh)
        reg.norms = input_norms
        projection = maps.Projection(self.mesh.n_cells, np.arange(self.mesh.n_cells))

        other_reg = regularization.WeightedLeastSquares(self.mesh)

        invProb = inverse_problem.BaseInvProblem(self.dmis, reg + other_reg, self.opt)

        beta_schedule = directives.BetaSchedule(coolingFactor=3)

        # Here is where the norms are applied
        irls_directive = directives.UpdateIRLS(
            cooling_factor=3,
            chifact_start=100.0,
            chifact_target=1.0,
            irls_cooling_factor=1.2,
            f_min_change=np.inf,
            max_irls_iterations=20,
            misfit_tolerance=1e-0,
            percentile=100,
            verbose=True,
        )

        assert irls_directive.cooling_factor == 3
        assert irls_directive.metrics is not None

        # TODO Move these assertion test to the 'test_validation_in_inversion' after update
        with self.assertRaises(AssertionError):
            inversion.BaseInversion(
                invProb, directiveList=[beta_schedule, irls_directive]
            )

        with self.assertRaises(AssertionError):
            inversion.BaseInversion(
                invProb, directiveList=[beta_schedule, irls_directive]
            )

        spherical_weights = directives.SphericalUnitsWeights(projection, [reg])
        with self.assertRaises(AssertionError):
            inversion.BaseInversion(
                invProb, directiveList=[irls_directive, spherical_weights]
            )

        update_Jacobi = directives.UpdatePreconditioner()
        with self.assertRaises(AssertionError):
            inversion.BaseInversion(
                invProb, directiveList=[update_Jacobi, irls_directive]
            )

        invProb.phi_d = 1.0
        self.opt.iter = 3
        invProb.model = np.random.randn(reg.regularization_mesh.nC)
        inv = inversion.BaseInversion(invProb, directiveList=[irls_directive])

        irls_directive.initialize()
        assert irls_directive.metrics.input_norms == [input_norms, None]
        assert reg.norms == [2.0, 2.0, 2.0, 2.0]

        irls_directive.inversion = inv
        irls_directive.endIter()

        assert irls_directive.metrics.start_irls_iter == self.opt.iter
        assert len(reg.objfcts[0]._weights) == 2  # With irls weights
        assert len(other_reg.objfcts[0]._weights) == 1  # No irls
        irls_directive.metrics.irls_iteration_count += 1
        irls_directive.endIter()

        assert self.opt.stopNextIteration

        # Test stopping criteria based on max_irls_iter
        irls_directive.max_irls_iterations = 2
        assert irls_directive.stopping_criteria()

        # Test beta re-adjustment down
        invProb.phi_d = 4.0
        irls_directive.misfit_tolerance = 0.1
        irls_directive.adjust_cooling_schedule()
        assert irls_directive.cooling_factor == 2.0

        # Test beta re-adjustment up
        invProb.phi_d = 0.5
        irls_directive.adjust_cooling_schedule()
        assert irls_directive.cooling_factor == 0.5

    def test_spherical_weights(self):
        reg = regularization.Sparse(self.mesh)
        projection = maps.Projection(self.mesh.n_cells, np.arange(self.mesh.n_cells))
        for obj in reg.objfcts[1:]:
            obj.units = "radian"

        with pytest.raises(TypeError, match="Attribute 'amplitude' must be of type"):
            directives.SphericalUnitsWeights(reg, [reg])

        with pytest.raises(TypeError, match="Attribute 'angles' must be a list of"):
            directives.SphericalUnitsWeights(projection, ["abc"])

        spherical_weights = directives.SphericalUnitsWeights(projection, [reg])

        inv_prob = inverse_problem.BaseInvProblem(self.dmis, reg, self.opt)
        model = np.abs(np.random.randn(reg.regularization_mesh.nC))
        inv_prob.model = model
        inv = inversion.BaseInversion(inv_prob, directiveList=[spherical_weights])
        spherical_weights.inversion = inv

        spherical_weights.initialize()

        assert "angle_scale" not in reg.objfcts[0]._weights
        assert reg.objfcts[1]._weights["angle_scale"].max() == model.max() / np.pi

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


class TestDeprecatedArguments:
    """
    Test if directives raise errors after passing deprecated arguments.
    """

    def test_debug(self):
        """
        Test if InversionDirective raises error after passing 'debug'.
        """
        msg = "'debug' property has been removed. Please use 'verbose'."
        with pytest.raises(TypeError, match=msg):
            directives.InversionDirective(debug=True)


class TestUpdateSensitivityWeightsRemovedArgs:
    """
    Test if `UpdateSensitivityWeights` raises errors after passing removed arguments.
    """

    def test_every_iter(self):
        """
        Test if `UpdateSensitivityWeights` raises error after passing `everyIter`.
        """
        msg = "'everyIter' property has been removed. Please use 'every_iteration'."
        with pytest.raises(TypeError, match=msg):
            directives.UpdateSensitivityWeights(everyIter=True)

    def test_threshold(self):
        """
        Test if `UpdateSensitivityWeights` raises error after passing `threshold`.
        """
        msg = "'threshold' property has been removed. Please use 'threshold_value'."
        with pytest.raises(TypeError, match=msg):
            directives.UpdateSensitivityWeights(threshold=True)

    def test_normalization(self):
        """
        Test if `UpdateSensitivityWeights` raises error after passing `normalization`.
        """
        msg = (
            "'normalization' property has been removed. "
            "Please define normalization using 'normalization_method'."
        )
        with pytest.raises(TypeError, match=msg):
            directives.UpdateSensitivityWeights(normalization=True)


class TestUpdateSensitivityNormalization:
    """
    Test the `normalization` property and setter in `UpdateSensitivityWeights`
    """

    @pytest.mark.parametrize("normalization_method", (None, "maximum", "minimum"))
    def test_normalization_method_setter_valid(self, normalization_method):
        """
        Test if the setter method for normalization_method in
        `UpdateSensitivityWeights` works as expected on valid values.

        The `normalization_method` must be a string or a None. This test was
        included as part of the removal process of the old `normalization`
        property.
        """
        d_temp = directives.UpdateSensitivityWeights()
        # Use the setter method to assign a value to normalization_method
        d_temp.normalization_method = normalization_method
        assert d_temp.normalization_method == normalization_method

    @pytest.mark.parametrize("normalization_method", (True, False, "an invalid method"))
    def test_normalization_method_setter_invalid(self, normalization_method):
        """
        Test if the setter method for normalization_method in
        `UpdateSensitivityWeights` raises error on invalid values.

        The `normalization_method` must be a string or a None. This test was
        included as part of the removal process of the old `normalization`
        property.
        """
        d_temp = directives.UpdateSensitivityWeights()
        if isinstance(normalization_method, bool):
            error_type = TypeError
            msg = "'normalization_method' must be a str. Got"
        else:
            error_type = ValueError
            msg = (
                r"'normalization_method' must be in \['minimum', 'maximum'\]. "
                f"Got '{normalization_method}'"
            )
        with pytest.raises(error_type, match=msg):
            d_temp.normalization_method = normalization_method


class TestSeedProperty:
    """
    Test ``seed`` setter methods of directives.
    """

    directive_classes = (
        directives.AlphasSmoothEstimate_ByEig,
        directives.BetaEstimate_ByEig,
        directives.BetaEstimateMaxDerivative,
        directives.ScalingMultipleDataMisfits_ByEig,
    )

    @pytest.mark.parametrize("directive_class", directive_classes)
    @pytest.mark.parametrize(
        "seed",
        (42, np.random.default_rng(seed=1), np.array([1, 2])),
        ids=("int", "rng", "array"),
    )
    def test_valid_seed(self, directive_class, seed):
        "Test if seed setter works as expected on valid seed arguments."
        directive = directive_class(seed=seed)
        assert directive.seed is seed

    @pytest.mark.parametrize("directive_class", directive_classes)
    @pytest.mark.parametrize("seed", (42.1, np.array([1.0, 2.0])))
    def test_invalid_seed(self, directive_class, seed):
        "Test if seed setter works as expected on valid seed arguments."
        msg = "Unable to initialize the random number generator with "
        with pytest.raises(TypeError, match=msg):
            directive_class(seed=seed)


class TestBetaEstimatorArguments:
    """
    Test if arguments are assigned in beta estimator directives.
    These tests catch the bug described and fixed in #1460.
    """

    def test_beta_estimate_by_eig(self):
        """Test on directives.BetaEstimate_ByEig."""
        beta0_ratio = 3.0
        n_pw_iter = 3
        seed = 42
        directive = directives.BetaEstimate_ByEig(
            beta0_ratio=beta0_ratio, n_pw_iter=n_pw_iter, seed=seed
        )
        assert directive.beta0_ratio == beta0_ratio
        assert directive.n_pw_iter == n_pw_iter
        assert directive.seed == seed

    def test_beta_estimate_max_derivative(self):
        """Test on directives.BetaEstimateMaxDerivative."""
        beta0_ratio = 3.0
        seed = 42
        directive = directives.BetaEstimateMaxDerivative(
            beta0_ratio=beta0_ratio, seed=seed
        )
        assert directive.beta0_ratio == beta0_ratio
        assert directive.seed == seed


if __name__ == "__main__":
    unittest.main()
