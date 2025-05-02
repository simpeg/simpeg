"""
Test how potential field surveys and simulations access receiver components.
"""

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


class TestMagneticSimulationDifferential:

    def build_survey(self, receivers: list | None):
        """
        Build a sample survey.
        """
        source_field = magnetics.sources.UniformBackgroundField(
            receiver_list=receivers, amplitude=55_000, inclination=12, declination=35
        )
        survey = magnetics.survey.Survey(source_field)
        return survey

    @pytest.fixture
    def sample_simulation(self, mesh, receiver_locations):
        """
        Build a sample simulation with single receiver with "tmi".
        """
        receivers = [magnetics.receivers.Point(receiver_locations, components="tmi")]
        source_field = magnetics.sources.UniformBackgroundField(
            receiver_list=receivers, amplitude=55_000, inclination=12, declination=35
        )
        survey = magnetics.survey.Survey(source_field)
        simulation = magnetics.Simulation3DDifferential(
            mesh, survey=survey, muMap=maps.IdentityMap(mesh=mesh)
        )
        return simulation

    def test_survey_setter(self, receiver_locations, sample_simulation):
        """
        Test ``survey`` setter with valid receivers.
        """
        receivers = [magnetics.receivers.Point(receiver_locations, components="tmi")]
        survey = self.build_survey(receivers)
        # Try to override the survey, should pass wo errors
        sample_simulation.survey = survey

    @pytest.mark.parametrize("invalid_rx", ["no-rx", "different-components"])
    def test_survey_setter_invalid(
        self, receiver_locations, sample_simulation, invalid_rx
    ):
        """
        Test ``survey`` setter with invalid receivers.
        """
        if invalid_rx == "no-rx":
            receivers = []
            msg = re.escape("Found invalid survey without receivers.")
        else:
            receivers = [
                magnetics.receivers.Point(receiver_locations, components=c)
                for c in ("tmi", ["bx", "by"])
            ]
            msg = re.escape(
                "Found invalid survey with receivers that have mixed components."
            )
        # Try to override the survey
        survey = self.build_survey(receivers)
        with pytest.raises(ValueError, match=msg):
            sample_simulation.survey = survey

    @pytest.mark.parametrize("components", ["tmi", ["bx", "by", "bz"]])
    def test_get_components(self, mesh, receiver_locations, components):
        """
        Test the ``_get_components`` method with valid receivers.
        """
        receivers = [
            magnetics.receivers.Point(receiver_locations, components=components),
            magnetics.receivers.Point(receiver_locations, components=components),
        ]
        survey = self.build_survey(receivers)
        simulation = magnetics.Simulation3DDifferential(
            mesh, survey=survey, muMap=maps.IdentityMap(mesh=mesh)
        )

        expected = components if isinstance(components, list) else [components]
        assert expected == simulation._get_components()

    @pytest.mark.parametrize("invalid_rx", ["no-rx", "different-components"])
    def test_get_components_invalid(
        self, sample_simulation, receiver_locations, invalid_rx
    ):
        """
        Test the ``_get_components`` with invalid receivers.
        """
        # Override receivers in simulation's survey
        if invalid_rx == "no-rx":
            receivers = []
            msg = re.escape("Found invalid survey without receivers.")
        else:
            receivers = [
                magnetics.receivers.Point(receiver_locations, components=c)
                for c in ("tmi", ["bx", "by"])
            ]
            msg = re.escape(
                "Found invalid survey with receivers that have mixed components."
            )
        # Override private attribute `_receiver_list` to bypass the setter
        sample_simulation.survey.source_field._receiver_list = receivers

        # Try to get components
        with pytest.raises(ValueError, match=msg):
            sample_simulation._get_components()

    @pytest.mark.parametrize("invalid_rx", ["no-rx", "different-components"])
    @pytest.mark.parametrize("method", ["projectFields", "projectFieldsDeriv"])
    def test_project_fields_invalid(
        self, sample_simulation, receiver_locations, invalid_rx, method
    ):
        """
        Test ``projectFields`` and ``projectFieldsDeriv`` on invalid surveys.
        """
        # Override receivers in simulation's survey
        if invalid_rx == "no-rx":
            receivers = None
            msg = re.escape("Found invalid survey without receivers.")
        else:
            receivers = [
                magnetics.receivers.Point(receiver_locations, components=c)
                for c in ("tmi", ["bx", "by", "bz"])
            ]
            msg = re.escape(
                "Found invalid survey with receivers that have mixed components."
            )
        # Override private attribute `_receiver_list` to bypass the setter
        sample_simulation.survey.source_field._receiver_list = receivers

        # Compute fields from a random model
        n_cells = sample_simulation.mesh.n_cells
        model = np.random.default_rng(seed=42).uniform(size=n_cells)
        fields = sample_simulation.fields(model)

        # Test errors
        method = getattr(sample_simulation, method)
        with pytest.raises(ValueError, match=msg):
            method(fields)
