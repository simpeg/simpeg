import unittest
import warnings
import pytest
import numpy as np

from SimPEG import (
    Mesh, Maps, Directives, Regularization, DataMisfit, Optimization,
    Inversion, InvProblem
)
from SimPEG import PF

np.random.seed(20)


def magInvSetup():
    mesh = Mesh.TensorMesh([8, 8, 8])

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
        mesh, chiMap=Maps.IdentityMap(mesh), silent=True
    )

    # Pair the survey and problem
    survey.pair(prob)

    # Compute forward model some data
    m = np.random.rand(mesh.nC)
    survey.makeSyntheticData(m)

    reg = Regularization.Sparse(mesh)
    reg.mref = np.zeros(mesh.nC)

    wr = np.sum(prob.F**2., axis=0)**0.5
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

    return mesh, invProb


class DirectivesValidation(unittest.TestCase):

    def test_validation_pass(self):
        betaest = Directives.BetaEstimate_ByEig()

        IRLS = Directives.Update_IRLS(
            f_min_change=1e-4, minGNiter=3, beta_tol=1e-2
        )
        update_Jacobi = Directives.Update_lin_PreCond()
        dList = [betaest, IRLS, update_Jacobi]
        directiveList = Directives.DirectiveList(*dList)

        self.assertTrue(directiveList.validate())

    def test_validation_fail(self):
        betaest = Directives.BetaEstimate_ByEig()

        IRLS = Directives.Update_IRLS(
            f_min_change=1e-4, minGNiter=3, beta_tol=1e-2
        )
        update_Jacobi = Directives.Update_lin_PreCond()
        dList = [betaest, update_Jacobi, IRLS]
        directiveList = Directives.DirectiveList(*dList)

        with self.assertRaises(AssertionError):
            self.assertTrue(directiveList.validate())

    def test_validation_warning(self):
        betaest = Directives.BetaEstimate_ByEig()

        IRLS = Directives.Update_IRLS(
            f_min_change=1e-4, minGNiter=3, beta_tol=1e-2
        )
        update_Jacobi = Directives.Update_lin_PreCond()
        dList = [betaest, IRLS]
        directiveList = Directives.DirectiveList(*dList)

        with pytest.warns(UserWarning):
            self.assertTrue(directiveList.validate())


class ValidationInInversion(unittest.TestCase):

    def setUp(self):
        mesh, invProb = magInvSetup()
        self.mesh = mesh
        self.invProb = invProb

    def test_validation_in_inversion(self):
        betaest = Directives.BetaEstimate_ByEig()

        # Here is where the norms are applied
        IRLS = Directives.Update_IRLS(
            f_min_change=1e-4, minGNiter=3, beta_tol=1e-2
        )

        update_Jacobi = Directives.Update_lin_PreCond()

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


class KillInversionAfterIter(Directives.InversionDirective):

    stopping_criteria_satisfied = False

    def endIter(self):
        self.stopping_criteria_satisfied = True


class ErrorOnIter(Directives.InversionDirective):

    def endIter(self):
        if self.opt.iter > 1:
            raise Exception


class StoppingCriteria(unittest.TestCase):

    def setUp(self):
        mesh, invProb = magInvSetup()
        self.mesh = mesh
        self.invProb = invProb

    def test_directive_stopping_None(self):
        betaest = Directives.BetaEstimate_ByEig()
        dList = [betaest]
        directiveList = Directives.DirectiveList(*dList)

        inv = Inversion.BaseInversion(
                self.invProb, directiveList=directiveList
            )

        # directives list shouldn't stop the inversion if the directive
        # has no stopping criteria
        self.assertFalse(directiveList.stopping_criteria_satisfied)

    def test_directive_stopping(self):
        betaest = Directives.BetaEstimate_ByEig()
        targetMisfit = Directives.TargetMisfit(phi_d_star=10.)
        dList = [betaest, targetMisfit]
        directiveList = Directives.DirectiveList(*dList)

        inv = Inversion.BaseInversion(
                self.invProb, directiveList=directiveList
            )
        self.invProb.startup(np.zeros(self.mesh.nC))

        # should be false to start
        self.assertFalse(directiveList.stopping_criteria_satisfied)

        # if we set the inv prob phi_d below phi_d_star, the stopping
        # criteria should be true
        self.invProb.phi_d = 3.
        self.assertTrue(directiveList.stopping_criteria_satisfied)

    def test_directive_stopping_multiple(self):
        betaest = Directives.BetaEstimate_ByEig()
        targetMisfit = Directives.TargetMisfit(phi_d_star=10.)
        targetMisfit2 = Directives.TargetMisfit(phi_d_star=2.)

        dList = [betaest, targetMisfit, targetMisfit2]
        directiveList = Directives.DirectiveList(*dList)

        inv = Inversion.BaseInversion(
                self.invProb, directiveList=directiveList
            )
        self.invProb.startup(np.zeros(self.mesh.nC))

        # should be false to start
        self.assertFalse(directiveList.stopping_criteria_satisfied)

        # if we set the inv prob phi_d below phi_d_star for target 1
        # stopping criteria should still be false
        self.invProb.phi_d = 3.
        self.assertFalse(directiveList.stopping_criteria_satisfied)

        # if we set the inv prob phi_d below phi_d_star for both targets
        # stopping criteria should still be true
        self.invProb.phi_d = 1.
        print(directiveList.stopping_criteria_satisfied)
        self.assertTrue(directiveList.stopping_criteria_satisfied)

    def test_stopping_executed(self):
        inv = Inversion.BaseInversion(
            self.invProb, directiveList=[
                Directives.BetaEstimate_ByEig(), ErrorOnIter()
            ]
        )

        # make sure we do run enough iterations by default to trigger exception
        with self.assertRaises(Exception):
            inv.run(np.zeros(self.mesh.nC))

        _, invProb2 = magInvSetup()
        inv = Inversion.BaseInversion(
                invProb2,
                directiveList=[
                    Directives.BetaEstimate_ByEig(), ErrorOnIter(),
                    KillInversionAfterIter()
                ]
            )
        inv.run(np.zeros(self.mesh.nC))



if __name__ == '__main__':
    unittest.main()
