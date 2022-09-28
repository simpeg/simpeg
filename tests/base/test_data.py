import unittest

import numpy as np
import numpy.testing as npt
import scipy.sparse as sp
import discretize

from SimPEG import maps, utils
from SimPEG import data_misfit, simulation, survey
from SimPEG import Data
from SimPEG.data import (
    _check_invalid_and_missing_components,
    _check_data_sizes,
    _observed_data_dict_to_array,
)

from SimPEG.potential_fields.gravity.receivers import Point
from SimPEG.potential_fields.gravity.sources import SourceField
from SimPEG.potential_fields.gravity.survey import Survey

np.random.seed(17)


class TestObservedDataDictToArray(unittest.TestCase):
    def setUp(self):
        """
        Create sample objects for the following tests
        """
        self.dobs = {
            "guv": np.array([2, 5, 8]),
            "gxy": np.array([3, 6, 9]),
            "gz": np.array([1, 4, 7]),
        }
        receiver_locations = np.array(
            [
                [0, 0, 100],
                [1, 1, 100],
                [-1, -1, 100],
            ],
            dtype=float,
        )
        components = ["gz", "guv", "gxy"]
        receivers = Point(receiver_locations, components=components)
        source_field = SourceField(receiver_list=[receivers])
        self.survey = Survey(source_field)

    def test_observed_data_dict_to_array(self):
        """
        Test if the conversion of dobs dict to array works properly
        """
        dobs_array = _observed_data_dict_to_array(
            self.dobs, self.survey.components.keys()
        )
        expected_array = np.linspace(1, 9, 9, dtype=int)
        npt.assert_allclose(dobs_array, expected_array)

    def test_data_class_with_dobs_as_dict(self):
        """
        Test SimPEG.Data when passing dobs as a dict
        """
        data = Data(self.survey, dobs=self.dobs)
        expected_array = np.linspace(1, 9, 9, dtype=int)
        npt.assert_allclose(data.dobs, expected_array)


class TestChecksForDobsAsDicts(unittest.TestCase):
    def test_check_data_sizes(self):
        """
        Check if _check_data_sizes doesn't raise error on valid dict
        """
        dobs = {
            "a": np.array([1, 4, 7]),
            "b": np.array([2, 5, 8]),
            "c": np.array([3, 6, 9]),
        }
        _check_data_sizes(dobs)

    def test_check_data_sizes_invalid(self):
        """
        Check if _check_data_sizes catch invalid sizes of data arrays
        """
        dobs = {
            "a": np.array([1, 4, 7]),
            "b": np.ones(10),
            "c": np.array([3, 6, 9]),
        }
        with self.assertRaises(ValueError):
            _check_data_sizes(dobs)

    def test_check_invalid_and_missing_components(self):
        """
        Check if _check_invalid_and_missing_components doesn't raise error on
        valid dict_keys
        """
        dobs = {
            "a": np.array([1, 4, 7]),
            "b": np.array([2, 5, 8]),
            "c": np.array([3, 6, 9]),
        }
        survey_components = {"c": None, "a": None, "b": None}
        _check_invalid_and_missing_components(dobs.keys(), survey_components.keys())

    def test_check_invalid_and_missing_components_invalid(self):
        """
        Check if _check_invalid_and_missing_components raises error on
        invalid dict_keys
        """
        dobs = {
            "z": np.array([1, 4, 7]),
            "b": np.array([2, 5, 8]),
            "c": np.array([3, 6, 9]),
        }
        survey_components = {"c": None, "a": None, "b": None}
        with self.assertRaises(ValueError):
            _check_invalid_and_missing_components(dobs.keys(), survey_components.keys())

    def test_check_invalid_and_missing_components_missing(self):
        """
        Check if _check_invalid_and_missing_components raises error on
        missing dict_keys
        """
        dobs = {
            "b": np.array([2, 5, 8]),
            "c": np.array([3, 6, 9]),
        }
        survey_components = {"c": None, "a": None, "b": None}
        with self.assertRaises(ValueError):
            _check_invalid_and_missing_components(dobs.keys(), survey_components.keys())


class DataTest(unittest.TestCase):
    def setUp(self):
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
        self.assertTrue(all(data.relative_error == relative * np.ones(len(self.dobs))))
        self.assertTrue(all(data.standard_deviation == relative * np.abs(self.dobs)))

    def test_instantiation_noise_floor(self):
        floor = np.min(np.abs(self.dobs))
        data = Data(self.sim.survey, dobs=self.dobs, noise_floor=floor)
        self.assertTrue(all(data.noise_floor == floor * np.ones(len(self.dobs))))
        self.assertTrue(all(data.standard_deviation == floor * np.ones(len(self.dobs))))

    def test_instantiation_relative_floor(self):
        relative = 0.5
        floor = np.min(np.abs(self.dobs))
        data = Data(
            self.sim.survey, dobs=self.dobs, relative_error=relative, noise_floor=floor
        )
        self.assertTrue(all(data.relative_error == relative * np.ones(len(self.dobs))))
        self.assertTrue(all(data.noise_floor == floor * np.ones(len(self.dobs))))
        self.assertTrue(
            np.allclose(
                data.standard_deviation,
                np.sqrt(
                    (relative * np.abs(self.dobs)) ** 2
                    + floor ** 2 * np.ones(len(self.dobs)),
                ),
            )
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

        self.assertTrue(all(data.noise_floor == standard_deviation))
        self.assertTrue(all(data.standard_deviation == standard_deviation))


if __name__ == "__main__":
    unittest.main()
