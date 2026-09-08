"""
Test receiver's ``eval`` method.
"""

import numpy as np
import pytest
from simpeg.electromagnetics import natural_source as nsem
from simpeg.electromagnetics.natural_source.utils.test_utils import setup1DSurvey
from simpeg.utils.solver_utils import get_default_solver


@pytest.mark.parametrize("orientation", ["xx", "yy"])
def test_zero_value(orientation):
    """
    Test if ``Impedance.eval()`` returns an array of zeros on 1D problem
    when orientation is ``"xx"`` or ``"yy"``.

    Test bugfix introduced in #1692.
    """
    survey, sigma, _, mesh = setup1DSurvey(sigmaHalf=1e-2, rx_orientation=orientation)

    # Define simulation and precompute fields
    solver = get_default_solver()
    simulation = nsem.Simulation1DPrimarySecondary(
        mesh, sigmaPrimary=sigma, sigma=sigma, survey=survey, solver=solver
    )
    fields = simulation.fields()

    # Check if calling eval on each receiver returns the expected result
    sources_and_receivers = (
        (src, rx) for src in survey.source_list for rx in src.receiver_list
    )
    for source, receiver in sources_and_receivers:
        result = receiver.eval(source, mesh, fields)
        np.testing.assert_allclose(result, 0)
        assert result.shape == (receiver.nD, 1)
