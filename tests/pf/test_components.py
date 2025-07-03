"""
Test how potential field surveys and simulations access receiver components.
"""

import re
import pytest
import numpy as np

import discretize
from simpeg.potential_fields import gravity, magnetics


@pytest.fixture
def receiver_locations():
    x = np.linspace(-20.0, 20.0, 4)
    x, y = np.meshgrid(x, x)
    z = 5.0 * np.ones_like(x)
    return np.vstack((x.ravel(), y.ravel(), z.ravel())).T


@pytest.fixture
def mesh():
    dh = 5.0
    hx = [(dh, 10)]
    return discretize.TensorMesh([hx, hx, hx], "CCN")


class TestComponentsGravitySurvey:

    def test_deprecated_components(self, receiver_locations):
        """
        Test FutureError after deprecated ``components`` property.
        """
        receivers = gravity.receivers.Point(receiver_locations, components="gz")
        source_field = gravity.sources.SourceField(receiver_list=[receivers])
        survey = gravity.survey.Survey(source_field)
        msg = re.escape("The `components` property is deprecated")
        with pytest.warns(FutureWarning, match=msg):
            survey.components


class TestComponentsMagneticSurvey:

    def test_deprecated_components(self, receiver_locations):
        """
        Test FutureError after deprecated ``components`` property.
        """
        receivers = magnetics.receivers.Point(receiver_locations, components="tmi")
        source_field = magnetics.sources.UniformBackgroundField(
            receiver_list=[receivers], amplitude=55_000, inclination=12, declination=35
        )
        survey = magnetics.survey.Survey(source_field)
        msg = re.escape("The `components` property is deprecated")
        with pytest.warns(FutureWarning, match=msg):
            survey.components
