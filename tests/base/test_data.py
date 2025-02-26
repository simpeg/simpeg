import re
import pytest
import unittest

import numpy as np
import discretize

from simpeg import maps
from simpeg import simulation, survey
from simpeg import Data


class DataTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(17)
        mesh = discretize.TensorMesh([30])
        sigma = np.ones(mesh.nC)
        model = np.log(sigma)

        receivers = survey.BaseRx(20 * [[0.0]])
        source = survey.BaseSrc([receivers])

        self.sim = simulation.ExponentialSinusoidSimulation(
            mesh=mesh, survey=survey.BaseSurvey([source]), model_map=maps.ExpMap(mesh)
        )

        self.dobs = self.sim.dpred(model)

    def test_instantiation_relative_error(self):
        relative = 0.5
        data = Data(self.sim.survey, dobs=self.dobs, relative_error=relative)
        np.testing.assert_equal(data.relative_error, relative * np.ones(len(self.dobs)))
        np.testing.assert_equal(data.standard_deviation, relative * np.abs(self.dobs))

    def test_instantiation_noise_floor(self):
        floor = np.min(np.abs(self.dobs))
        data = Data(self.sim.survey, dobs=self.dobs, noise_floor=floor)
        np.testing.assert_equal(data.noise_floor, floor * np.ones(len(self.dobs)))
        np.testing.assert_equal(
            data.standard_deviation, floor * np.ones(len(self.dobs))
        )

    def test_instantiation_relative_floor(self):
        relative = 0.5
        floor = np.min(np.abs(self.dobs))
        data = Data(
            self.sim.survey, dobs=self.dobs, relative_error=relative, noise_floor=floor
        )
        np.testing.assert_equal(data.relative_error, relative * np.ones(len(self.dobs)))
        np.testing.assert_equal(data.noise_floor, floor * np.ones(len(self.dobs)))
        np.testing.assert_allclose(
            data.standard_deviation,
            np.sqrt(
                (relative * np.abs(self.dobs)) ** 2
                + floor**2 * np.ones(len(self.dobs)),
            ),
        )

    def test_instantiation_standard_deviation(self):
        relative = 0.5
        floor = np.min(np.abs(self.dobs))
        standard_deviation = relative * np.abs(self.dobs) + floor * np.ones(
            len(self.dobs)
        )
        data = Data(
            self.sim.survey, dobs=self.dobs, standard_deviation=standard_deviation
        )

        np.testing.assert_equal(data.noise_floor, standard_deviation)
        np.testing.assert_equal(data.standard_deviation, standard_deviation)


class TestNansInData:
    """
    Test if errors are raised after passing arrays with nans to the ``Data`` object.
    """

    @pytest.fixture
    def n_data(self):
        return 4

    @pytest.fixture
    def sample_survey(self, n_data):
        receivers = survey.BaseRx(np.zeros(n_data)[:, np.newaxis])
        source = survey.BaseSrc([receivers])
        return survey.BaseSurvey([source])

    @pytest.fixture
    def array(self, n_data):
        return np.ones(n_data, dtype=np.float64)

    @pytest.fixture
    def array_with_nan(self, array):
        array = array.copy()
        array[1] = np.nan
        return array

    @pytest.mark.parametrize("method", ["init", "setter"])
    def test_dobs(self, array, array_with_nan, sample_survey, method):
        """
        Test errors when passing arrays to the constructor.
        """
        expected_msg = re.escape("Invalid 'dobs' with nan values.")
        if method == "init":
            with pytest.raises(ValueError, match=expected_msg):
                Data(sample_survey, dobs=array_with_nan)
        else:
            data = Data(sample_survey, dobs=array)
            with pytest.raises(ValueError, match=expected_msg):
                data.dobs = array_with_nan

    @pytest.mark.parametrize("method", ["init", "setter"])
    @pytest.mark.parametrize(
        "nan_property",
        [
            "relative_error",
            "noise_floor",
            "standard_deviation",
        ],
    )
    def test_uncertanties(
        self, array, array_with_nan, sample_survey, method, nan_property
    ):
        """
        Test errors when passing arrays to the constructor.
        """
        expected_msg = re.escape(f"Invalid '{nan_property}' with nan values.")
        if method == "init":
            kwargs = {nan_property: array_with_nan}
            with pytest.raises(ValueError, match=expected_msg):
                Data(sample_survey, dobs=array, **kwargs)
        else:
            data = Data(sample_survey, dobs=array)
            with pytest.raises(ValueError, match=expected_msg):
                setattr(data, nan_property, array_with_nan)
