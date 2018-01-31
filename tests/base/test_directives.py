import unittest
import warnings
import pytest
import numpy as np

from SimPEG import (
    Mesh, Maps, Directives, Regularization, DataMisfit, Optimization,
    Inversion, InvProblem
)
from SimPEG import PF


class DirectivesValidation(unittest.TestCase):

    def test_validation_pass(self):
        betaest = Directives.BetaEstimate_ByEig()

        IRLS = Directives.Update_IRLS(
            f_min_change=1e-4, minGNiter=3, beta_tol=1e-2
        )
        update_Jacobi = Directives.UpdatePreconditioner()
        dList = [betaest, IRLS, update_Jacobi]
        directiveList = Directives.DirectiveList(*dList)

        self.assertTrue(directiveList.validate())

    def test_validation_fail(self):
        betaest = Directives.BetaEstimate_ByEig()

        IRLS = Directives.Update_IRLS(
            f_min_change=1e-4, minGNiter=3, beta_tol=1e-2
        )
        update_Jacobi = Directives.UpdatePreconditioner()
        dList = [betaest, update_Jacobi, IRLS]
        directiveList = Directives.DirectiveList(*dList)

        with self.assertRaises(AssertionError):
            self.assertTrue(directiveList.validate())

    def test_validation_warning(self):
        betaest = Directives.BetaEstimate_ByEig()

        IRLS = Directives.Update_IRLS(
            f_min_change=1e-4, minGNiter=3, beta_tol=1e-2
        )
        update_Jacobi = Directives.UpdatePreconditioner()
        dList = [betaest, IRLS]
        directiveList = Directives.DirectiveList(*dList)

        with pytest.warns(UserWarning):
            self.assertTrue(directiveList.validate())


class ValidationInInversion(unittest.TestCase):

    def setUp(self):
        mesh = Mesh.TensorMesh([4, 4, 4])

        # Magnetic inducing field parameter (A,I,D)
        B = [50000, 90, 0]

        # Create a MAGsurvey
        rx = PF.BaseMag.RxObs(
            np.vstack([[0.25, 0.25, 0.25], [-0.25, -0.25, 0.25]])
        )
        srcField = PF.BaseMag.SrcField([rx], param=(B[0], B[1], B[2]))
        survey = PF.BaseMag.LinearSurvey(srcField)

        # Create the forward model operator
        prob = PF.Magnetics.MagneticIntegral(
            mesh, chiMap=Maps.IdentityMap(mesh)
        )

        # Pair the survey and problem
        survey.pair(prob)

        # Compute forward model some data
        m = np.random.rand(mesh.nC)
        survey.makeSyntheticData(m)

        reg = Regularization.Sparse(mesh)
        reg.mref = np.zeros(mesh.nC)

        wr = np.sum(prob.G**2., axis=0)**0.5
        reg.cell_weights = wr
        reg.norms = [0, 1, 1, 1]
        reg.eps_p, reg.eps_q = 1e-3, 1e-3

        # Data misfit function
        dmis = DataMisfit.l2_DataMisfit(survey)
        dmis.W = 1./survey.std

        # Add directives to the inversion
        opt = Optimization.ProjectedGNCG(
            maxIter=2, lower=-10., upper=10.,
            maxIterCG=2
        )

        invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

        self.mesh = mesh
        self.invProb = invProb

    def test_validation_in_inversion(self):
        betaest = Directives.BetaEstimate_ByEig()

        # Here is where the norms are applied
        IRLS = Directives.Update_IRLS(
            f_min_change=1e-4, minGNiter=3, beta_tol=1e-2
        )

        update_Jacobi = Directives.UpdatePreconditioner()

        with self.assertRaises(AssertionError):
            # validation should happen and this will fail
            # (IRLS needs to be before update_Jacobi)
            inv = Inversion.BaseInversion(
                self.invProb, directiveList=[betaest, update_Jacobi, IRLS]
            )

        with self.assertRaises(AssertionError):
            # validation should happen and this will fail
            # (IRLS needs to be before update_Jacobi)
            inv = Inversion.BaseInversion(self.invProb)
            inv.directiveList = [betaest, update_Jacobi, IRLS]


if __name__ == '__main__':
    unittest.main()
