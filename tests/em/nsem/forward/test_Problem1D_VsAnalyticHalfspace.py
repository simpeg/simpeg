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
        mesh, survey=survey, sigmaPrimary=sigBG, sigma=sigma,
    )

    # Get the fields
    fields = simulation.fields()

    # Project the data
    data = simulation.dpred(f=fields)

    # Calculate the app res and phs
    app_r = np.array(nsem.utils.test_utils.getAppResPhs(data, survey=survey))[:, 0]

    return app_r


def appPhs_psFieldNorm(sigmaHalf):

    # Make the survey
    survey, sigma, sigBG, mesh = nsem.utils.test_utils.setup1DSurvey(sigmaHalf, False)
    simulation = nsem.Simulation1DPrimarySecondary(
        mesh, survey=survey, sigmaPrimary=sigBG, sigma=sigma
    )

    # Get the fields
    fields = simulation.fields()

    # Project the data
    data = simulation.dpred(f=fields)

    # Calculate the app  phs
    app_p = np.array(nsem.utils.test_utils.getAppResPhs(data, survey))[:, 1]

    return app_p


class TestAnalytics(unittest.TestCase):
    def setUp(self):
        pass

    # Primary/secondary
    def test_appRes1en0_ps(self):
        sigma_half = 1.0
        np.testing.assert_allclose(
            appRes_psFieldNorm(sigma_half), 1 / sigma_half, rtol=TOLr
        )

    def test_appPhs1en0_ps(self):
        sigma_half = 1.0
        np.testing.assert_allclose(appPhs_psFieldNorm(sigma_half), -135, rtol=TOLp)

    def test_appRes2en1_ps(self):
        sigma_half = 2e-1
        np.testing.assert_allclose(
            appRes_psFieldNorm(sigma_half), 1 / sigma_half, rtol=TOLr
        )

    def test_appPhs2en1_ps(self):
        sigma_half = 2e-1
        np.testing.assert_allclose(appPhs_psFieldNorm(sigma_half), -135, rtol=TOLp)

    def test_appRes2en3_ps(self):
        sigma_half = 2e-3
        np.testing.assert_allclose(
            appRes_psFieldNorm(sigma_half), 1 / sigma_half, rtol=TOLr
        )

    def test_appPhs2en3_ps(self):
        sigma_half = 2e-1
        np.testing.assert_allclose(appPhs_psFieldNorm(sigma_half), -135, rtol=TOLp)


if __name__ == "__main__":
    unittest.main()
