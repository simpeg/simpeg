import re
from collections import namedtuple
from datetime import datetime
import pathlib
import unittest
import warnings
from statistics import harmonic_mean

import pytest
import numpy as np

import discretize
import simpeg
from simpeg import (
    maps,
    directives,
    regularization,
    optimization,
    inversion,
    inverse_problem,
    simulation,
)
from simpeg.data_misfit import L2DataMisfit
from simpeg.potential_fields import gravity
from simpeg.potential_fields import magnetics as mag
import shutil

from simpeg.regularization.base import Smallness
from simpeg.regularization.sparse import Sparse


class directivesValidation(unittest.TestCase):

    def test_error_irls_and_beta_scheduling(self):
        """
        Test if validation error when ``UpdateIRLS`` and ``BetaSchedule`` are present.
        """
        directives_list = directives.DirectiveList(
            directives.UpdateIRLS(),
            directives.BetaSchedule(coolingFactor=2, coolingRate=1),
        )
        msg = "Beta scheduling is handled by the"
        with pytest.raises(AssertionError, match=msg):
            directives_list.validate()

    def test_validation_pass(self):
        betaest = directives.BetaEstimate_ByEig()

        IRLS = directives.UpdateIRLS()

        update_Jacobi = directives.UpdatePreconditioner()
        dList = [betaest, IRLS, update_Jacobi]
        directiveList = directives.DirectiveList(*dList)

        self.assertTrue(directiveList.validate())

    def test_validation_fail(self):
        betaest = directives.BetaEstimate_ByEig()

        IRLS = directives.UpdateIRLS()
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

        IRLS = directives.UpdateIRLS()
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

        data = sim.make_synthetic_data(m, add_noise=True, random_seed=19)
        dmis = L2DataMisfit(data=data, simulation=sim)
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
        IRLS = directives.UpdateIRLS(f_min_change=1e-4)
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

        expected_target = self.dmis.nD
        # Test beta re-adjustment down
        invProb.phi_d = 4.0
        irls_directive.misfit_tolerance = 0.1
        irls_directive.adjust_cooling_schedule()

        ratio = invProb.phi_d / expected_target
        expected_factor = harmonic_mean([4 / 3, ratio])
        np.testing.assert_allclose(irls_directive.cooling_factor, expected_factor)

        # Test beta re-adjustment up
        invProb.phi_d = 1 / 2
        ratio = invProb.phi_d / expected_target
        expected_factor = harmonic_mean([1 / 2, ratio])

        irls_directive.adjust_cooling_schedule()
        np.testing.assert_allclose(irls_directive.cooling_factor, expected_factor)

        # Test beta no-adjustment
        irls_directive.cooling_factor = (
            2.0  # set this to something not 1 to make sure it changes to 1.
        )

        invProb.phi_d = expected_target * (
            1 + irls_directive.misfit_tolerance * 0.5
        )  # something within the relative tolerance
        irls_directive.adjust_cooling_schedule()
        assert irls_directive.cooling_factor == 1

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
    data = sim.make_synthetic_data(
        np.ones(mesh.n_cells), add_noise=True, random_seed=20
    )
    dmis = L2DataMisfit(data, sim)

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


class TestRandomSeedProperty:
    """
    Test ``random_seed`` setter methods of directives.
    """

    directive_classes = (
        directives.AlphasSmoothEstimate_ByEig,
        directives.BetaEstimate_ByEig,
        directives.BetaEstimateMaxDerivative,
        directives.ScalingMultipleDataMisfits_ByEig,
    )

    @pytest.mark.parametrize("directive_class", directive_classes)
    @pytest.mark.parametrize(
        "random_seed",
        (42, np.random.default_rng(seed=1), np.array([1, 2])),
        ids=("int", "rng", "array"),
    )
    def test_valid_seed(self, directive_class, random_seed):
        "Test if seed setter works as expected on valid seed arguments."
        directive = directive_class(random_seed=random_seed)
        assert directive.random_seed is random_seed

    @pytest.mark.parametrize("directive_class", directive_classes)
    @pytest.mark.parametrize("random_seed", (42.1, np.array([1.0, 2.0])))
    def test_invalid_seed(self, directive_class, random_seed):
        "Test if seed setter works as expected on valid seed arguments."
        msg = "Unable to initialize the random number generator with "
        with pytest.raises(TypeError, match=msg):
            directive_class(random_seed=random_seed)


class TestBetaEstimatorArguments:
    """
    Test if arguments are assigned in beta estimator directives.
    These tests catch the bug described and fixed in #1460.
    """

    def test_beta_estimate_by_eig(self):
        """Test on directives.BetaEstimate_ByEig."""
        beta0_ratio = 3.0
        n_pw_iter = 3
        random_seed = 42
        directive = directives.BetaEstimate_ByEig(
            beta0_ratio=beta0_ratio, n_pw_iter=n_pw_iter, random_seed=random_seed
        )
        assert directive.beta0_ratio == beta0_ratio
        assert directive.n_pw_iter == n_pw_iter
        assert directive.random_seed == random_seed

    def test_beta_estimate_max_derivative(self):
        """Test on directives.BetaEstimateMaxDerivative."""
        beta0_ratio = 3.0
        random_seed = 42
        directive = directives.BetaEstimateMaxDerivative(
            beta0_ratio=beta0_ratio, random_seed=random_seed
        )
        assert directive.beta0_ratio == beta0_ratio
        assert directive.random_seed == random_seed


class TestRemovedSeedProperty:
    """
    Test removal of seed property.
    """

    CLASSES = (
        directives.AlphasSmoothEstimate_ByEig,
        directives.BetaEstimate_ByEig,
        directives.BetaEstimateMaxDerivative,
        directives.ScalingMultipleDataMisfits_ByEig,
    )

    def get_message_removed_error(self, old_name, new_name, version="v0.24.0"):
        msg = (
            f"'{old_name}' has been removed in "
            f" SimPEG {version}, please use '{new_name}' instead."
        )
        return msg

    @pytest.mark.parametrize("directive", CLASSES)
    def test_error_argument(self, directive):
        """
        Test if error is raised after passing ``seed`` to the constructor.
        """
        msg = self.get_message_removed_error("seed", "random_seed")
        with pytest.raises(TypeError, match=msg):
            directive(seed=42135)

    @pytest.mark.parametrize("directive", CLASSES)
    def test_error_accessing_property(self, directive):
        """
        Test error when trying to access the ``seed`` property.
        """
        directive_obj = directive(random_seed=42)
        msg = "seed has been removed, please use random_seed"
        with pytest.raises(NotImplementedError, match=msg):
            directive_obj.seed


class TestUpdateIRLS:
    """
    Additional tests to UpdateIRLS directive.
    """

    @pytest.fixture
    def mesh(self):
        """Sample tensor mesh."""
        return discretize.TensorMesh([4, 4, 4])

    @pytest.fixture
    def data_misfit(self, mesh):
        rx = mag.Point(np.vstack([[0.25, 0.25, 0.25], [-0.25, -0.25, 0.25]]))
        igrf = mag.UniformBackgroundField(
            receiver_list=[rx], amplitude=5000, inclination=90, declination=0
        )
        survey = mag.Survey(igrf)
        sim = mag.Simulation3DIntegral(
            mesh, survey=survey, chiMap=maps.IdentityMap(mesh)
        )
        model = np.random.default_rng(seed=42).normal(size=mesh.n_cells)
        data = sim.make_synthetic_data(model, add_noise=True)
        dmisfit = L2DataMisfit(data=data, simulation=sim)
        return dmisfit

    def test_end_iter_irls_threshold(self, mesh, data_misfit):
        """
        Test if irls_threshold is modified in every regularization term after
        the IRLS process started.
        """
        # Define a regularization combo with sparse and non-sparse terms
        irls_threshold = 4.5
        sparse_regularization = Sparse(
            mesh, norms=[1, 1, 1, 1], irls_threshold=irls_threshold
        )
        non_sparse_regularization = Smallness(mesh)
        reg = 0.1 * sparse_regularization + 0.5 * non_sparse_regularization
        # Define inversion
        opt = optimization.ProjectedGNCG()
        opt.iter = 0  # manually set iter to zero
        inv_prob = inverse_problem.BaseInvProblem(data_misfit, reg, opt)
        inv_prob.phi_d = np.nan  # manually set value for phi_d
        inv_prob.model = np.zeros(mesh.n_cells)  # manually set the model
        # Define inversion
        inv = inversion.BaseInversion(inv_prob)
        irls_cooling_factor = 1.2
        update_irls = directives.UpdateIRLS(
            irls_cooling_factor=irls_cooling_factor,
            inversion=inv,
            dmisfit=data_misfit,
            reg=reg,
        )
        # Modify metrics to kick in the IRLS process
        update_irls.metrics.start_irls_iter = 0
        # Check irls_threshold of the objective function terms after running endIter
        update_irls.endIter()
        for obj_fun in sparse_regularization.objfcts:
            assert obj_fun.irls_threshold == irls_threshold / irls_cooling_factor
        # The irls_threshold for the sparse_regularization should not be changed
        assert sparse_regularization.irls_threshold == irls_threshold


class DummySaveEveryIteration(directives.SaveEveryIteration):
    """
    Dummy non-abstract class to test SaveEveryIteration.
    """

    @property
    def file_abs_path(self) -> pathlib.Path:
        """
        Simple implementation of abstract property file_abs_path.
        """
        return self.directory / self.name


class MockOpt:
    """Mock Opt object."""

    def __init__(self, xc=None, maxIter=100):
        if xc is None:
            xc = np.random.default_rng(seed=42).uniform(size=23)
        self.xc = xc
        self.maxIter = maxIter


class MockInvProb:
    """Mock InvProb object."""

    def __init__(self, opt):
        self.opt = opt


class MockInversion:
    """Mock Inversion object."""

    def __init__(self, xc=None, maxIter=100):
        opt = MockOpt(xc=xc, maxIter=maxIter)
        inv_prob = MockInvProb(opt)
        self.invProb = inv_prob


class TestSaveEveryIteration:
    """Test the SaveEveryIteration directive."""

    @pytest.mark.parametrize("directory", ["dummy/path", "../dummy/path"])
    def test_directory(self, directory):
        """Test the directory property."""
        directive = DummySaveEveryIteration(directory=directory)
        assert directive.directory == pathlib.Path(directory).resolve()

    def test_no_directory(self):
        """Test if the directory property is None when on_disk is False"""
        directive = DummySaveEveryIteration(directory="blah", on_disk=False)
        assert directive._directory is None

        # accessing the directive property should raise error when on_disk is False
        msg = re.escape("directory' is only available")
        with pytest.raises(AttributeError, match=msg):
            directive.directory

        # using the directive setter should raise error when on_disk is False

    @pytest.mark.parametrize("directory", ["dummy/path", "../dummy/path"])
    def test_directory_setter(self, directory):
        """Test the directory setter."""
        directive = DummySaveEveryIteration()
        directive.directory = directory
        assert directive.directory == pathlib.Path(directory).resolve()

    def test_directory_setter_error_none(self):
        """Test error when trying to set directory=None if on_disk is True."""
        directive = DummySaveEveryIteration()
        msg = re.escape("Directory is not optional if 'on_disk==True'")
        with pytest.raises(ValueError, match=msg):
            directive.directory = None

    def test_name(self):
        """Test the name property."""
        name = "blah"
        directive = DummySaveEveryIteration(name=name)
        assert directive.name == name

    def test_name_setter(self):
        """Test the name setter."""
        directive = DummySaveEveryIteration()
        name = "blah"
        directive.name = name
        assert directive.name == name

    def test_mkdir(self, tmp_path):
        """Test _mkdir_and_check_output_file."""
        directory = tmp_path / "blah"
        directive = DummySaveEveryIteration(directory=directory)
        directive._mkdir_and_check_output_file()
        assert directory.exists()
        fname = directory / directive.name
        assert not fname.exists()

    @pytest.mark.parametrize(
        "should_exist", [True, False], ids=["should_exist", "should_not_exist"]
    )
    def test_check_output_file_exists(self, tmp_path, should_exist):
        """Test _mkdir_and_check_output_file when file exists."""
        directory = tmp_path / "blah"
        directory.mkdir(parents=True)
        directive = DummySaveEveryIteration(directory=directory)
        fname = directive.file_abs_path
        fname.touch()
        assert fname.exists()

        if should_exist:
            # No warning should be raised if exists and should exist
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                directive._mkdir_and_check_output_file(should_exist=should_exist)
        else:
            # Warning should be raised if exists and should not exist
            with pytest.warns(UserWarning, match="Overwriting file"):
                directive._mkdir_and_check_output_file(should_exist=should_exist)

    @pytest.mark.parametrize(
        "should_exist", [True, False], ids=["should_exist", "should_not_exist"]
    )
    def test_check_output_file_doesnt_exist(self, tmp_path, should_exist):
        """Test _mkdir_and_check_output_file when file doesn't exist."""
        directory = tmp_path / "blah"
        directory.mkdir(parents=True)
        directive = DummySaveEveryIteration(directory=directory)
        fname = directive.file_abs_path

        if should_exist:
            # Warning should be raised if doesn't exist and should exist
            with pytest.warns(
                UserWarning, match=re.escape(f"File {fname} was not found")
            ):
                directive._mkdir_and_check_output_file(should_exist=should_exist)
        else:
            # No warning should be raised if doesn't exist and should not exist
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                directive._mkdir_and_check_output_file(should_exist=should_exist)

    @pytest.mark.parametrize("opt", [True, False], ids=["with-opt", "without-opt"])
    def test_initialize(self, opt):
        """
        Test the initialize method.
        """
        directive = DummySaveEveryIteration()
        if opt:
            directive.inversion = MockInversion(maxIter=10000)

        expected_start_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
        directive.initialize()
        assert directive._start_time == expected_start_time

        if opt:
            # maxIter was set to 10000, so the _iter_format should be "05d"
            assert directive._iter_format == "05d"

    def test_time_iter_no_opt(self):
        directive = DummySaveEveryIteration(name="dummy")
        time_name = directive._time_file_name.name
        assert directive._time_iter_file_name.name == time_name + "_###"

    def test_deprecated_fileName(self):
        directive = DummySaveEveryIteration(name="dummy")

        with pytest.warns(FutureWarning, match=r"'fileName' has been deprecated .*"):
            f_name = directive.fileName

        assert f_name == "dummy"


class TestSaveModelEveryIteration:
    """Test the SaveModelEveryIteration directive."""

    def test_on_disk(self):
        """
        Test on_disk is always True.
        """
        directive = directives.SaveModelEveryIteration()
        assert directive.on_disk

    def test_on_disk_argument(self):
        """
        Test warning after passing on_disk as argument.
        """
        msg = re.escape("The 'on_disk' argument is ignored")
        with pytest.warns(UserWarning, match=msg):
            directive = directives.SaveModelEveryIteration(on_disk=False)
        assert directive.on_disk

    def test_on_disk_setter(self):
        """
        Test error after trying to modify value of on_disk.
        """
        directive = directives.SaveModelEveryIteration()
        msg = re.escape("Cannot modify value of 'on_disk'")
        with pytest.raises(AttributeError, match=msg):
            directive.on_disk = False

    def test_end_iter(self, tmp_path):
        """
        Test if endIter saves the model to a file.
        """
        directory = tmp_path / "dummy_dir"
        directive = directives.SaveModelEveryIteration(directory=directory)

        # Add a mock inversion to the directive
        mock_inversion = MockInversion()
        directive.inversion = mock_inversion

        # Initialize and call endIter
        directive.initialize()
        directive.endIter()

        # Check if file exists
        assert directory.exists()
        assert directive.file_abs_path.exists()
        array = np.load(directive.file_abs_path)

        np.testing.assert_equal(array, mock_inversion.invProb.opt.xc)


class BaseTestOutputDirective:
    """
    Base class to test directives that need a full inversion.
    """

    def get_inversion_problem(self):
        """
        Simple gravity inversion problem to test the directive.
        """
        # Mesh
        # ----
        h = [(1.0, 6)]
        mesh = discretize.TensorMesh([h, h, h], origin="CCN")

        # Survey
        # ------
        x = np.linspace(-2.0, 2.0, 5)
        xx, yy = np.meshgrid(x, x)
        zz = 1.0 * np.ones_like(xx)
        receiver_locations = np.vstack([c.ravel() for c in (xx, yy, zz)]).T
        receivers = gravity.Point(locations=receiver_locations, components="gz")
        source_field = gravity.SourceField([receivers])
        survey = gravity.Survey(source_field)

        # Simulation
        # ----------
        mapping = simpeg.maps.IdentityMap(mesh=mesh)
        simulation = gravity.Simulation3DIntegral(
            mesh=mesh, survey=survey, rhoMap=mapping, engine="choclo"
        )

        # Synthetic data
        # --------------
        model = np.zeros(mesh.n_cells)
        model = simpeg.utils.model_builder.add_block(
            mesh.cell_centers,
            model,
            p0=[-1.0, -1.0, -2.0],
            p1=[1.0, 1.0, -1.0],
            prop_value=200,
        )
        synthetic_data = simulation.make_synthetic_data(
            model,
            relative_error=0.1,
            random_seed=4,
            add_noise=True,
        )

        # Inversion problem
        # -----------------
        data_misfit = simpeg.data_misfit.L2DataMisfit(
            data=synthetic_data, simulation=simulation
        )
        regularization = simpeg.regularization.WeightedLeastSquares(mesh)
        optimizer = optimization.ProjectedGNCG()
        inv_prob = simpeg.inverse_problem.BaseInvProblem(
            data_misfit, regularization, optimizer
        )

        return inv_prob

    def get_directives(self, save_output_directive: directives.SaveEveryIteration):
        """
        Get list of directives to use in the sample gravity inversion.

        Include the save_output_directive passed as argument in the list.
        """
        sensitivity_weights = simpeg.directives.UpdateSensitivityWeights(
            every_iteration=False
        )
        update_jacobi = simpeg.directives.UpdatePreconditioner(
            update_every_iteration=True
        )
        starting_beta = simpeg.directives.BetaEstimate_ByEig(beta0_ratio=10)
        beta_schedule = simpeg.directives.BetaSchedule(coolingFactor=2.0, coolingRate=1)
        target_misfit = simpeg.directives.TargetMisfit(chifact=1.0)

        directives_list = [
            sensitivity_weights,
            starting_beta,
            update_jacobi,
            beta_schedule,
            save_output_directive,
            target_misfit,
        ]
        return directives_list


class TestSaveOutputEveryIteration(BaseTestOutputDirective):
    """
    Test the SaveOutputEveryIteration directive.
    """

    @pytest.mark.parametrize("on_disk", [True, False])
    def test_initialize(self, tmp_path, on_disk):
        """Test the initialize method."""
        directory = tmp_path / "dummy"
        directive = directives.SaveOutputEveryIteration(
            on_disk=on_disk, directory=directory
        )
        directive.initialize()

        # Check directory was created
        if on_disk:
            assert directory.exists()

            # Check that the file was created
            assert directive.file_abs_path is not None
            assert directive.file_abs_path.exists()

            # Check header in file
            with directive.file_abs_path.open(mode="r") as f:
                lines = f.readlines()
            assert len(lines) == 1
            assert "beta" in lines[0]
            assert "phi_d" in lines[0]
            assert "phi_m" in lines[0]
        else:
            assert directive.file_abs_path is None

        assert directive.beta == []
        assert directive.phi_d == []
        assert directive.phi_m == []
        assert directive.phi_m_smooth_z == []
        assert directive.phi == []

    @pytest.mark.parametrize(
        ("on_disk", "test_load_results"),
        [
            pytest.param(
                True,
                True,
                marks=pytest.mark.xfail(
                    reason="bug in load_results", raises=AttributeError
                ),
            ),
            (True, False),
            (False, None),
        ],
        ids=["on_disk-test_load_results", "on_disk", "not_on_disk"],
    )
    def test_end_iter(self, tmp_path, on_disk, test_load_results):
        """Test the endIter method."""
        inv_prob = self.get_inversion_problem()

        directory = tmp_path / "dummy"
        directive = directives.SaveOutputEveryIteration(
            directory=directory, on_disk=on_disk
        )
        directives_list = self.get_directives(directive)
        inversion = simpeg.inversion.BaseInversion(inv_prob, directives_list)

        initial_model = np.zeros(inv_prob.dmisfit.nP)
        inversion.run(initial_model)

        # Check that lists are not empty
        lists = [
            "beta",
            "phi_d",
            "phi_m",
            "phi_m_small",
            "phi_m_smooth_x",
            "phi_m_smooth_y",
            "phi_m_smooth_z",
            "phi",
        ]
        for attribute in lists:
            assert getattr(directive, attribute)

        # Just exit the test if on_disk is False
        if not on_disk:
            return

        # Check that the file was created if on_disk
        assert directive.file_abs_path is not None
        assert directive.file_abs_path.exists()

        # Check content of file
        with directive.file_abs_path.open(mode="r") as f:
            lines = f.readlines()
        assert "beta" in lines[0]
        assert "phi_d" in lines[0]
        assert "phi_m" in lines[0]
        assert len(lines) > 1

        # Test load_results
        if test_load_results:
            original_values = {attr: getattr(directive, attr) for attr in lists}
            for attribute in lists:
                # Clean the lists
                setattr(directive, attribute, [])

            # Load results and check if they are the same as the original ones
            directive.load_results()

            # Check that the original values were recovered
            for attribute in lists:
                np.testing.assert_equal(
                    getattr(directive, attribute),
                    original_values[attribute],
                )

    def test_load_results_error(self, tmp_path):
        """
        Test error when no file_name is passed to load_results.
        """
        directory = tmp_path / "dummy"
        directive = directives.SaveOutputEveryIteration(
            directory=directory, on_disk=False
        )
        msg = re.escape("'file_name' is a required argument")
        with pytest.raises(TypeError, match=msg):
            directive.load_results()


class TestSaveOutputDictEveryIteration(BaseTestOutputDirective):
    """
    Test the SaveOutputDictEveryIteration directive.
    """

    def test_initialize(self):
        """Test the initialize method."""
        directive = directives.SaveOutputDictEveryIteration()
        directive.initialize()

        # Check outDict was created and is empty
        assert hasattr(directive, "outDict")
        assert not directive.outDict

    @pytest.mark.parametrize("on_disk", [True, False], ids=["on_disk", "not_on_disk"])
    def test_end_iter(self, tmp_path, on_disk):
        """Test the endIter method."""
        inv_prob = self.get_inversion_problem()

        directory = tmp_path / "dummy"
        directive = directives.SaveOutputDictEveryIteration(
            directory=directory, on_disk=on_disk
        )
        directives_list = self.get_directives(directive)
        inversion = simpeg.inversion.BaseInversion(inv_prob, directives_list)

        initial_model = np.zeros(inv_prob.dmisfit.nP)
        inversion.run(initial_model)

        # Check if the outDict is not empty
        assert directive.outDict
        fields = [
            "iter",
            "beta",
            "phi_d",
            "phi_m",
            "f",
            "m",
            "dpred",
        ]
        for iteration in directive.outDict:
            for field in fields:
                assert field in directive.outDict[iteration]

        # Check if output files were created
        if on_disk:
            assert directory.exists()
            assert directory.is_dir()
            files = list(directory.iterdir())
            assert len(files) == len(directive.outDict)

    def test_deprecated(self):
        with pytest.warns(FutureWarning, match=".*saveOnDisk has been deprecated.*"):
            directive = directives.SaveOutputDictEveryIteration(saveOnDisk=True)

        assert directive.on_disk

    @pytest.mark.parametrize("on_disk", [True, False], ids=["on_disk", "not_on_disk"])
    def test_file_abs_path_optional(self, on_disk):
        directive = directives.SaveOutputDictEveryIteration(on_disk=on_disk)
        if on_disk:
            assert directive.file_abs_path is not None
        else:
            assert directive.file_abs_path is None


class MockJointInvProb:

    def __init__(self):
        self.opt = namedtuple("Opt", "f iter cg_count")(0.1, 10, 200)
        self.betas = [1, 2, 3]
        self.phi_d_list = [0.1, 0.2, 0.3]
        self.phi_m_list = [0.2, 0.3, 0.4]
        self.lambd = 1e5
        self.phi_sim = 10


class TestSimMeasureSaveOutputEveryIteration:

    @pytest.mark.parametrize("on_disk", [True, False])
    def test_initialize(self, tmp_path, on_disk):
        """Test the initialize method."""
        directory = tmp_path / "dummy"
        directive = directives.SimilarityMeasureSaveOutputEveryIteration(
            on_disk=on_disk, directory=directory
        )
        directive.initialize()

        # Check directory was created
        if on_disk:
            assert directory.exists()

            # Check that the file was created
            assert directive.file_abs_path is not None
            assert directive.file_abs_path.exists()

            # Check header in file
            with directive.file_abs_path.open(mode="r") as f:
                lines = f.readlines()
            assert len(lines) == 1
            assert "betas" in lines[0]
            assert "joint_phi_d" in lines[0]
            assert "joint_phi_m" in lines[0]
            assert "phi_sim" in lines[0]
        else:
            assert directive.file_abs_path is None

        assert directive.betas == []
        assert directive.lambd == []
        assert directive.phi_d == []
        assert directive.phi_m == []
        assert directive.phi_sim == []
        assert directive.phi == []

    @pytest.mark.parametrize("on_disk", [True, False])
    def test_end_iter(self, tmp_path, on_disk):
        directory = tmp_path / "dummy"
        directive = directives.SimilarityMeasureSaveOutputEveryIteration(
            on_disk=on_disk, directory=directory
        )
        directive.initialize()

        joint_problem = MockJointInvProb()
        joint_inv = namedtuple("JointInversion", "invProb")(joint_problem)
        directive.inversion = joint_inv

        assert directive.invProb is joint_inv.invProb

        directive.endIter()

        assert directive.betas == [joint_problem.betas]
        assert directive.phi_d == [joint_problem.phi_d_list]
        assert directive.phi_m == [joint_problem.phi_m_list]
        assert directive.lambd == [joint_problem.lambd]
        assert directive.phi_sim == [joint_problem.phi_sim]
        assert directive.phi == [joint_problem.opt.f]

        if on_disk:
            out_file = directive.file_abs_path
            assert out_file.exists()

            n_lines = 0
            with open(out_file) as f:
                while f.readline():
                    n_lines += 1
            assert n_lines == 2  # header plus one line

    @pytest.mark.xfail(
        reason="np.loadtxt will not work to read in log file that has nested lists."
    )
    @pytest.mark.parametrize("pass_file_name", [True, False])
    def test_load_results(self, tmp_path, pass_file_name):
        directory = tmp_path / "dummy"
        directive = directives.SimilarityMeasureSaveOutputEveryIteration(
            directory=directory, on_disk=True
        )
        directive.initialize()

        joint_problem = MockJointInvProb()
        joint_inv = namedtuple("JointInversion", "invProb")(joint_problem)
        directive.inversion = joint_inv

        directive.endIter()

        if pass_file_name:
            log_file = directive.file_abs_path
            directive.load_results(log_file)
        else:
            directive.load_results()

        assert directive.betas == [joint_problem.betas]
        assert directive.phi_d == [joint_problem.phi_d_list]
        assert directive.phi_m == [joint_problem.phi_m_list]
        assert directive.lambd == [joint_problem.lambd]
        assert directive.phi_sim == [joint_problem.phi_sim]
        assert directive.phi == [joint_problem.opt.f]

    def test_load_results_error(self):
        directive = directives.SimilarityMeasureSaveOutputEveryIteration(on_disk=False)
        with pytest.raises(TypeError, match=r"'file_name' is a required argument.*"):
            directive.load_results()


if __name__ == "__main__":
    unittest.main()
