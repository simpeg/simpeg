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

    def test_sensitivity_weighting(self):

        tests_list = [
            {"everyIter": False, "threshold": 1e-12, "normalization": False},
            {
                "every_iteration": True,
                "threshold_value": 1,
                "threshold_method": "percentile",
                "normalization": True,
            },
            {
                "every_iteration": True,
                "threshold_value": 1e-3,
                "threshold_method": "amplitude",
                "normalization_method": "minimum",
            },
        ]

        # Test setter warnings
        d_temp = directives.UpdateSensitivityWeights()
        d_temp.normalization_method = True
        self.assertTrue(d_temp.normalization_method == "maximum")

        d_temp.normalization_method = False
        self.assertTrue(d_temp.normalization_method is None)

        # Compute test cell weights
        sqrt_diagJtJ = (
            np.sqrt(np.sum((self.dmis.W * self.sim.G) ** 2, axis=0))
            / self.mesh.cell_volumes
        )

        w1 = sqrt_diagJtJ + tests_list[0]["threshold"]  # default global thresholding
        w2 = np.clip(
            sqrt_diagJtJ,
            a_min=np.percentile(sqrt_diagJtJ, tests_list[1]["threshold_value"]),
            a_max=np.inf,
        )
        w2 /= w2.max()
        w3 = np.clip(
            sqrt_diagJtJ,
            a_min=tests_list[2]["threshold_value"] * sqrt_diagJtJ.max(),
            a_max=np.inf,
        )
        w3 /= w3.min()
        weights_list = [self.mesh.cell_volumes * w for w in [w1, w2, w3]]

        for ii, wi in enumerate(weights_list):

            reg = regularization.WeightedLeastSquares(self.mesh)
            invProb = inverse_problem.BaseInvProblem(self.dmis, reg, self.opt)
            invProb.model = self.model

            sensitivity_weights = directives.UpdateSensitivityWeights(**tests_list[ii])
            sensitivity_weights.inversion = inversion.BaseInversion(
                invProb, directiveList=[sensitivity_weights]
            )

            sensitivity_weights.update()

            for reg_i in reg.objfcts:
                self.assertTrue(np.all(np.isclose(wi, reg_i.cell_weights)))
                reg_i.remove_weights("sensitivity")

    def tearDown(self):
        # Clean up the working directory
        try:
            shutil.rmtree(self.sim.sensitivity_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    unittest.main()
