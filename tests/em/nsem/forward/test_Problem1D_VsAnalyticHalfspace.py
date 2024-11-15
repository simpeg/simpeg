import unittest
from simpeg.electromagnetics import natural_source as nsem
import numpy as np

# Define the tolerances
TOLr = 5e-1
TOLp = 5e-1


def appRes_psFieldNorm(conductivityHalf):
    # Make the survey
    survey, conductivity, sigBG, mesh = nsem.utils.test_utils.setup1DSurvey(
        conductivityHalf, False
    )
    simulation = nsem.Simulation1DPrimarySecondary(
        mesh,
        survey=survey,
        conductivityPrimary=sigBG,
        conductivity=conductivity,
    )

    # Get the fields
    fields = simulation.fields()

    # Project the data
    data = simulation.dpred(f=fields)

    # Calculate the app res and phs
    app_r = np.array(nsem.utils.test_utils.getAppResPhs(data, survey=survey))[:, 0]

    return app_r


def appPhs_psFieldNorm(conductivityHalf):
    # Make the survey
    survey, conductivity, sigBG, mesh = nsem.utils.test_utils.setup1DSurvey(
        conductivityHalf, False
    )
    simulation = nsem.Simulation1DPrimarySecondary(
        mesh, survey=survey, conductivityPrimary=sigBG, conductivity=conductivity
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
        conductivity_half = 1.0
        np.testing.assert_allclose(
            appRes_psFieldNorm(conductivity_half), 1 / conductivity_half, rtol=TOLr
        )

    def test_appPhs1en0_ps(self):
        conductivity_half = 1.0
        np.testing.assert_allclose(
            appPhs_psFieldNorm(conductivity_half), -135, rtol=TOLp
        )

    def test_appRes2en1_ps(self):
        conductivity_half = 2e-1
        np.testing.assert_allclose(
            appRes_psFieldNorm(conductivity_half), 1 / conductivity_half, rtol=TOLr
        )

    def test_appPhs2en1_ps(self):
        conductivity_half = 2e-1
        np.testing.assert_allclose(
            appPhs_psFieldNorm(conductivity_half), -135, rtol=TOLp
        )

    def test_appRes2en3_ps(self):
        conductivity_half = 2e-3
        np.testing.assert_allclose(
            appRes_psFieldNorm(conductivity_half), 1 / conductivity_half, rtol=TOLr
        )

    def test_appPhs2en3_ps(self):
        conductivity_half = 2e-1
        np.testing.assert_allclose(
            appPhs_psFieldNorm(conductivity_half), -135, rtol=TOLp
        )


if __name__ == "__main__":
    unittest.main()
