"""
Test how potential field surveys and simulations access receiver components.
"""

# Things to test:
#   - receivers with different components in the same survey
#       - projectFields
#       - projectFieldsDeriv
#       - _get_components
#   - deprecation error

import re
import pytest
import numpy as np

import discretize
from simpeg import maps
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


class TestComponentsMagneticSimulation:

    def build_simluation_differential(self, mesh, receivers: list | None):
        """
        Build a sample simulation object.
        """
        source_field = magnetics.sources.UniformBackgroundField(
            receiver_list=receivers, amplitude=55_000, inclination=12, declination=35
        )
        survey = magnetics.survey.Survey(source_field)
        simulation = magnetics.Simulation3DDifferential(
            mesh, survey=survey, muMap=maps.IdentityMap(mesh=mesh)
        )
        return simulation

    @pytest.mark.parametrize("components", ["tmi", ["bx", "by", "bz"]])
    def test_get_components(self, mesh, receiver_locations, components):
        receivers = [
            magnetics.receivers.Point(receiver_locations, components=components),
            magnetics.receivers.Point(receiver_locations, components=components),
        ]
        simulation = self.build_simluation_differential(mesh, receivers)

        expected = components if isinstance(components, list) else [components]
        assert expected == simulation._get_components()

    def test_get_components_no_receivers(self, mesh):
        simulation = self.build_simluation_differential(mesh, receivers=None)
        msg = re.escape("Found invalid survey without receivers.")
        with pytest.raises(ValueError, match=msg):
            simulation._get_components()

    @pytest.mark.parametrize(
        "method", ["_get_components", "projectFields", "projectFieldsDeriv"]
    )
    def test_different_components(self, mesh, receiver_locations, method):
        """
        Test NotImplementedError when receivers have different components.
        """
        # Define receivers with different components
        receivers = [
            magnetics.receivers.Point(receiver_locations, components="tmi"),
            magnetics.receivers.Point(
                receiver_locations, components=["bx", "by", "bz"]
            ),
        ]
        simulation = self.build_simluation_differential(mesh, receivers)

        # Compute fields from a random model
        model = np.random.default_rng(seed=42).uniform(size=mesh.n_cells)
        fields = simulation.fields(model)

        # Test NotImplementedErrors
        if method == "_get_components":
            with pytest.raises(NotImplementedError, match=""):
                simulation._get_components()
        else:
            msg = re.escape(
                "Found receivers with different set of components in the survey."
            )
            method = getattr(simulation, method)
            with pytest.raises(NotImplementedError, match=msg):
                method(fields)
