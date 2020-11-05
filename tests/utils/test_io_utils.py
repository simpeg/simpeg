from __future__ import print_function
import unittest
import numpy as np

from SimPEG.data import Data
from SimPEG.potential_fields import gravity, magnetics
from SimPEG.utils.io_utils import (
    write_gravity_3d_ubc,
    read_gravity_3d_ubc,
    write_gravity_gradiometry_3d_ubc,
    read_gravity_gradiometry_3d_ubc,
    write_magnetics_3d_ubc,
    read_magnetics_3d_ubc,
)
import os


# =============================================================
#                   POTENTIAL FIELDS
# =============================================================

print("=================================")
print("      TESTING GRAVITY IO")
print("=================================")


class TestGravityIO(unittest.TestCase):
    def setUp(self):

        np.random.seed(8)
        x = np.random.uniform(0, 100, 5)
        y = np.random.uniform(0, 100, 5)
        z = np.random.uniform(0, 100, 5)
        dobs = np.random.uniform(0, 10, 5)
        std = np.random.uniform(1, 10, 5)

        xyz = np.c_[x, y, z]
        receiver_list = [gravity.receivers.Point(xyz, components="gz")]
        source_field = gravity.sources.SourceField(receiver_list=receiver_list)
        survey = gravity.survey.Survey(source_field)

        self.survey = survey
        self.dobs = dobs
        self.std = std

    def test_io_survey(self):

        data_object = Data(survey=self.survey)
        filename = "survey.grv"

        write_gravity_3d_ubc(filename, data_object)
        data_loaded = read_gravity_3d_ubc(filename)
        os.remove(filename)

        passed = np.all(
            np.isclose(
                self.survey.receiver_locations, data_loaded.survey.receiver_locations
            )
        )
        self.assertTrue(passed, True)

        print("SURVEY FILE IO FOR GRAV3D PASSED")

    def test_io_dpred(self):

        data_object = Data(survey=self.survey, dobs=self.dobs)
        filename = "dpred.grv"

        write_gravity_3d_ubc(filename, data_object)
        data_loaded = read_gravity_3d_ubc(filename)
        os.remove(filename)

        passed = np.all(
            np.isclose(
                np.c_[self.survey.receiver_locations, self.dobs],
                np.c_[data_loaded.survey.receiver_locations, data_loaded.dobs],
            )
        )
        self.assertTrue(passed, True)

        print("PREDICTED DATA FILE IO FOR GRAV3D PASSED")

    def test_io_dobs(self):

        data_object = Data(
            survey=self.survey, dobs=self.dobs, standard_deviation=self.std
        )
        filename = "dpred.grv"

        write_gravity_3d_ubc(filename, data_object)
        data_loaded = read_gravity_3d_ubc(filename)
        os.remove(filename)

        passed = np.all(
            np.isclose(
                np.c_[self.survey.receiver_locations, self.dobs, self.std],
                np.c_[
                    data_loaded.survey.receiver_locations,
                    data_loaded.dobs,
                    data_loaded.standard_deviation,
                ],
            )
        )
        self.assertTrue(passed, True)

        print("OBSERVED DATA FILE IO FOR GRAV3D PASSED")


print("=================================")
print(" TESTING GRAVITY GRADIOMETRY IO")
print("=================================")


class TestGravityGradiometryIO(unittest.TestCase):
    def setUp(self):

        np.random.seed(8)
        x = np.random.uniform(0, 100, 5)
        y = np.random.uniform(0, 100, 5)
        z = np.random.uniform(0, 100, 5)
        dobs = np.random.uniform(0, 100, 6 * 5)
        std = np.random.uniform(1, 10, 6 * 5)

        components = ["gxx", "gxy", "gxz", "gyy", "gyz", "gzz"]
        xyz = np.c_[x, y, z]
        receiver_list = [gravity.receivers.Point(xyz, components=components)]
        source_field = gravity.sources.SourceField(receiver_list=receiver_list)
        survey = gravity.survey.Survey(source_field)

        self.survey = survey
        self.dobs = dobs
        self.std = std

    def test_io_survey(self):

        data_object = Data(survey=self.survey)
        filename = "survey.gg"

        write_gravity_gradiometry_3d_ubc(filename, data_object)
        data_loaded = read_gravity_gradiometry_3d_ubc(filename, "survey")
        os.remove(filename)

        passed = np.all(
            np.isclose(
                self.survey.receiver_locations, data_loaded.survey.receiver_locations
            )
        )
        self.assertTrue(passed, True)

        print("SURVEY FILE IO FOR GG3D PASSED")

    def test_io_dpred(self):

        data_object = Data(survey=self.survey, dobs=self.dobs)
        filename = "dpred.gg"

        write_gravity_gradiometry_3d_ubc(filename, data_object)
        data_loaded = read_gravity_gradiometry_3d_ubc(filename, "dpred")
        os.remove(filename)

        passed = np.all(
            np.isclose(
                self.survey.receiver_locations, data_loaded.survey.receiver_locations
            )
        )
        self.assertTrue(passed, True)

        passed = np.all(np.isclose(self.dobs, data_loaded.dobs))
        self.assertTrue(passed, True)

        print("PREDICTED DATA FILE IO FOR GG3D PASSED")

    def test_io_dobs(self):

        data_object = Data(
            survey=self.survey, dobs=self.dobs, standard_deviation=self.std
        )
        filename = "dpred.gg"

        write_gravity_gradiometry_3d_ubc(filename, data_object)
        data_loaded = read_gravity_gradiometry_3d_ubc(filename, "dobs")
        os.remove(filename)

        passed = np.all(
            np.isclose(
                self.survey.receiver_locations, data_loaded.survey.receiver_locations
            )
        )
        self.assertTrue(passed, True)

        passed = np.all(np.isclose(self.dobs, data_loaded.dobs))
        self.assertTrue(passed, True)

        passed = np.all(np.isclose(self.std, data_loaded.standard_deviation))
        self.assertTrue(passed, True)

        print("OBSERVED DATA FILE IO FOR GG3D PASSED")


print("=================================")
print("    TESTING MAGNETICS IO")
print("=================================")


class TestMagneticsIO(unittest.TestCase):
    def setUp(self):

        np.random.seed(8)
        x = np.random.uniform(0, 100, 5)
        y = np.random.uniform(0, 100, 5)
        z = np.random.uniform(0, 100, 5)
        dobs = np.random.uniform(0, 10, 5)
        std = np.random.uniform(1, 10, 5)

        xyz = np.c_[x, y, z]
        receiver_list = [magnetics.receivers.Point(xyz, components="tmi")]

        inducing_field = (50000.0, 60.0, 15.0)
        source_field = magnetics.sources.SourceField(
            receiver_list=receiver_list, parameters=inducing_field
        )
        survey = gravity.survey.Survey(source_field)

        self.survey = survey
        self.dobs = dobs
        self.std = std

    def test_io_survey(self):

        data_object = Data(survey=self.survey)
        filename = "survey.mag"

        write_magnetics_3d_ubc(filename, data_object)
        data_loaded = read_magnetics_3d_ubc(filename)
        os.remove(filename)

        passed = np.all(
            np.isclose(
                self.survey.receiver_locations, data_loaded.survey.receiver_locations
            )
        )
        self.assertTrue(passed, True)

        passed = np.all(
            np.isclose(
                self.survey.source_field.parameters,
                data_loaded.survey.source_field.parameters,
            )
        )
        self.assertTrue(passed, True)

        print("SURVEY FILE IO FOR MAG3D PASSED")

    def test_io_dpred(self):

        data_object = Data(survey=self.survey, dobs=self.dobs)
        filename = "dpred.mag"

        write_magnetics_3d_ubc(filename, data_object)
        data_loaded = read_magnetics_3d_ubc(filename)
        os.remove(filename)

        passed = np.all(
            np.isclose(
                np.c_[self.survey.receiver_locations, self.dobs],
                np.c_[data_loaded.survey.receiver_locations, data_loaded.dobs],
            )
        )
        self.assertTrue(passed, True)

        passed = np.all(
            np.isclose(
                self.survey.source_field.parameters,
                data_loaded.survey.source_field.parameters,
            )
        )
        self.assertTrue(passed, True)

        print("PREDICTED DATA FILE IO FOR MAG3D PASSED")

    def test_io_dobs(self):

        data_object = Data(
            survey=self.survey, dobs=self.dobs, standard_deviation=self.std
        )
        filename = "dpred.mag"

        write_magnetics_3d_ubc(filename, data_object)
        data_loaded = read_magnetics_3d_ubc(filename)
        os.remove(filename)

        passed = np.all(
            np.isclose(
                np.c_[self.survey.receiver_locations, self.dobs, self.std],
                np.c_[
                    data_loaded.survey.receiver_locations,
                    data_loaded.dobs,
                    data_loaded.standard_deviation,
                ],
            )
        )
        self.assertTrue(passed, True)

        passed = np.all(
            np.isclose(
                self.survey.source_field.parameters,
                data_loaded.survey.source_field.parameters,
            )
        )
        self.assertTrue(passed, True)

        print("OBSERVED DATA FILE IO FOR MAG3D PASSED")


if __name__ == "__main__":
    unittest.main()
