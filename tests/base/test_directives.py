import unittest
import warnings
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
        update_Jacobi = directives.UpdatePreconditioner()
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
        srcField = mag.SourceField([rx], parameters=(B[0], B[1], B[2]))
        survey = mag.Survey(srcField)

        # Create the forward model operator
        sim = mag.Simulation3DIntegral(
            mesh, survey=survey, chiMap=maps.IdentityMap(mesh)
        )

        # Compute forward model some data
        m = np.random.rand(mesh.nC)
        data = sim.make_synthetic_data(m, add_noise=True)

        reg = regularization.Sparse(mesh)
        reg.mref = np.zeros(mesh.nC)
        reg.norms = [0, 1, 1, 1]
        reg.eps_p, reg.eps_q = 1e-3, 1e-3

        # Data misfit function
        dmis = data_misfit.L2DataMisfit(data=data, simulation=sim)
        dmis.W = 1.0 / data.relative_error

        # Add directives to the inversion
        opt = optimization.ProjectedGNCG(
            maxIter=2, lower=-10.0, upper=10.0, maxIterCG=2
        )

        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)

        self.mesh = mesh
        self.invProb = invProb
        self.sim = sim

    def test_validation_in_inversion(self):
        betaest = directives.BetaEstimate_ByEig()

        # Here is where the norms are applied
        IRLS = directives.Update_IRLS(f_min_change=1e-4, minGNiter=3, beta_tol=1e-2)

        update_Jacobi = directives.UpdatePreconditioner()
        sensitivity_weights = directives.UpdateSensitivityWeights()
        with self.assertRaises(AssertionError):
            # validation should happen and this will fail
            # (IRLS needs to be before update_Jacobi)
            inv = inversion.BaseInversion(
                self.invProb, directiveList=[betaest, update_Jacobi, IRLS]
            )

        with self.assertRaises(AssertionError):
            # validation should happen and this will fail
            # (sensitivity_weights needs to be before betaest)
            inv = inversion.BaseInversion(
                self.invProb, directiveList=[betaest, sensitivity_weights]
            )

        with self.assertRaises(AssertionError):
            # validation should happen and this will fail
            # (sensitivity_weights needs to be before update_Jacobi)
            inv = inversion.BaseInversion(self.invProb)
            inv.directiveList = [update_Jacobi, sensitivity_weights]

    def tearDown(self):
        # Clean up the working directory
        try:
            shutil.rmtree(self.sim.sensitivity_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    unittest.main()
