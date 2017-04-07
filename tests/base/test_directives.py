import unittest
import warnings
import pytest
from SimPEG import Mesh, Directives


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


if __name__ == '__main__':
    unittest.main()
