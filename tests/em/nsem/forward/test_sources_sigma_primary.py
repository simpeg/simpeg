"""
Tests for the primary conductivity of the NSEM plane wave sources.
"""

import numpy as np
import discretize

from simpeg.electromagnetics import natural_source as nsem


def test_scalar_sigma_primary():
    h = [(100.0, 3, -1.5), (100.0, 6), (100.0, 3, 1.5)]
    mesh = discretize.TensorMesh([h, h, h], "CCC")
    survey = nsem.survey.Survey([nsem.sources.PlanewaveXYPrimary([], 1.0)])
    sim = nsem.simulation.Simulation3DPrimarySecondary(
        mesh, survey=survey, sigma=1e-2, sigmaPrimary=1e-2
    )
    sigma_1d, sigma_p = survey.source_list[0]._get_sigmas(sim)
    np.testing.assert_allclose(sigma_1d, 1e-2)
    np.testing.assert_allclose(sigma_p, 1e-2)
