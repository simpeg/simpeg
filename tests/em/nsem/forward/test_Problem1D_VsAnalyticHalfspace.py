import unittest
from SimPEG.electromagnetics import natural_source as nsem
import numpy as np

# Define the tolerances
TOLr = 5e-1
TOLp = 5e-1


def appRes_psFieldNorm(sigmaHalf):

    # Make the survey
    survey, sigma, sigBG, mesh = nsem.utils.test_utils.setup1DSurvey(sigmaHalf, False)
    simulation = nsem.Simulation1DPrimarySecondary(
        mesh, sigmaPrimary=sigBG, sigma=sigma
    )
    simulation.pair(survey)

    # Get the fields
    fields = simulation.fields()

    # Project the data
    data = simulation.dpred(f=fields)

    # Calculate the app res and phs
    app_r = np.array(nsem.utils.test_utils.getAppResPhs(data, survey=survey))[:, 0]

    return np.linalg.norm(
        np.abs(np.log(app_r) - np.log(np.ones(survey.nFreq) / sigmaHalf))
        * np.log(sigmaHalf)
    )


def appPhs_psFieldNorm(sigmaHalf):

    # Make the survey
    survey, sigma, sigBG, mesh = nsem.utils.test_utils.setup1DSurvey(sigmaHalf, False)
    simulation = nsem.Simulation1DPrimarySecondary(
        mesh, sigmaPrimary=sigBG, sigma=sigma
    )
    simulation.pair(survey)

    # Get the fields
    fields = simulation.fields()

    # Project the data
    data = simulation.dpred(f=fields)

    # Calculate the app  phs
    app_p = np.array(nsem.utils.test_utils.getAppResPhs(data, survey))[:, 1]

    return np.linalg.norm(np.abs(app_p - np.ones(survey.nFreq) * 45) / 45)


class TestAnalytics(unittest.TestCase):
    def setUp(self):
        pass

    # Primary/secondary
    def test_appRes1en0_ps(self):
        self.assertLess(appRes_psFieldNorm(1e-0), TOLr)

    def test_appPhs1en0_ps(self):
        self.assertLess(appPhs_psFieldNorm(1e-0), TOLp)

    def test_appRes2en1_ps(self):
        self.assertLess(appRes_psFieldNorm(2e-1), TOLr)

    def test_appPhs2en1_ps(self):
        self.assertLess(appPhs_psFieldNorm(2e-1), TOLp)

    def test_appRes2en3_ps(self):
        self.assertLess(appRes_psFieldNorm(2e-3), TOLr)

    def test_appPhs2en3_ps(self):
        self.assertLess(appPhs_psFieldNorm(2e-3), TOLp)


if __name__ == "__main__":
    unittest.main()
