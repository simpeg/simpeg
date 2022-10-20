from __future__ import print_function
import unittest
import numpy as np

from SimPEG.data import Data
from SimPEG.potential_fields import gravity, magnetics
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.utils.io_utils import (
    write_grav3d_ubc,
    read_grav3d_ubc,
    write_gg3d_ubc,
    read_gg3d_ubc,
    write_mag3d_ubc,
    read_mag3d_ubc,
    write_dcip3d_ubc,
    read_dcip3d_ubc,
    write_dcipoctree_ubc,
    read_dcipoctree_ubc,
)
import os


####################################################################################
#                                  POTENTIAL FIELDS
####################################################################################

print("==========================================")
print("           TESTING GRAVITY IO")
print("==========================================")


class TestIO_GRAV3D(unittest.TestCase):
    """
    A class for testing the read/write for UBC grav3d formatted data files.
    """

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

        write_grav3d_ubc(filename, data_object)
        data_loaded = read_grav3d_ubc(filename)
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

        write_grav3d_ubc(filename, data_object)
        data_loaded = read_grav3d_ubc(filename)
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

        write_grav3d_ubc(filename, data_object)
        data_loaded = read_grav3d_ubc(filename)
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


print("==========================================")
print("     TESTING GRAVITY GRADIOMETRY IO")
print("==========================================")


class TestIO_GG3D(unittest.TestCase):
    """
    A class for testing the read/write for UBC gg3d formatted data files.
    """

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

        write_gg3d_ubc(filename, data_object)
        data_loaded = read_gg3d_ubc(filename)
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

        write_gg3d_ubc(filename, data_object)
        data_loaded = read_gg3d_ubc(filename)
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

        write_gg3d_ubc(filename, data_object)
        data_loaded = read_gg3d_ubc(filename)
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


print("==========================================")
print("         TESTING MAGNETICS IO")
print("==========================================")


class TestIO_MAG3D(unittest.TestCase):
    """
    A class for testing the read/write for UBC mag3d formatted data files.
    """

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
        survey = magnetics.survey.Survey(source_field)

        self.survey = survey
        self.dobs = dobs
        self.std = std

    def test_io_survey(self):

        data_object = Data(survey=self.survey)
        filename = "survey.mag"

        write_mag3d_ubc(filename, data_object)
        data_loaded = read_mag3d_ubc(filename)
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

        write_mag3d_ubc(filename, data_object)
        data_loaded = read_mag3d_ubc(filename)
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

        write_mag3d_ubc(filename, data_object)
        data_loaded = read_mag3d_ubc(filename)
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


####################################################################################
#                        ELECTROMAGNETICS (STATICS)
####################################################################################

print("==========================================")
print("            TESTING DCIP IO")
print("==========================================")


class TestIO_DCIP3D(unittest.TestCase):
    """
    A class for testing the read/write for UBC dcip3d and dcipoctree formatted data files.
    """

    def setUp(self):

        # Receiver locations
        np.random.seed(8)
        xm = np.array([40.0, 50.0, 60.0])
        xn = np.array([70.0, 80.0, 90.0])
        ym = np.random.uniform(-5, 5, len(xm))
        zm = np.random.randn(len(xm))
        m_locs = np.c_[xm, ym, zm]
        n_locs = np.c_[xn, ym, zm]

        # Source locations
        np.random.seed(9)
        xa = np.array([0.0, 10.0])
        xb = np.array([20.0, 30.0])
        ya = np.random.uniform(-5, 5, len(xa))
        za = np.random.randn(len(xa))
        a_locs = np.c_[xa, ya, za]
        b_locs = np.c_[xb, ya, za]

        n_src = len(xa)
        n_rx = len(xm)

        # Define survey
        pp_sources = []
        dpdp_sources = []

        for ii in range(0, n_src):

            pp_receivers = [dc.receivers.Pole(m_locs)]
            dpdp_receivers = [dc.receivers.Dipole(m_locs, n_locs)]

            a_loc = a_locs[ii, :]
            b_loc = b_locs[ii, :]

            pp_sources.append(dc.sources.Pole(pp_receivers, a_loc))
            dpdp_sources.append(dc.sources.Dipole(dpdp_receivers, a_loc, b_loc))

        self.pp_survey = dc.survey.Survey(pp_sources, survey_type="pole-pole")
        self.dpdp_survey = dc.survey.Survey(dpdp_sources, survey_type="dipole-dipole")

        # Define data and uncertainties. In this case nD = 6
        n_data = len(xa) * len(xm)

        np.random.seed(10)
        dobs = np.random.uniform(1e-3, 1e-2, n_data)
        std = np.random.uniform(1e-5, 1e-4, n_data)

        self.dobs = dobs
        self.std = std

    def test_io_survey(self):

        pp_data = Data(survey=self.pp_survey)
        dpdp_data = Data(survey=self.dpdp_survey)

        filename = "survey.dc"

        # Test for pole-pole
        write_dcip3d_ubc(
            filename, pp_data, "secondary_potential", "survey", format_type="general"
        )
        data_loaded = read_dcip3d_ubc(filename, "secondary_potential")
        os.remove(filename)

        A = np.c_[
            self.pp_survey.locations_a,
            self.pp_survey.locations_b,
            self.pp_survey.locations_m,
            self.pp_survey.locations_n,
        ]

        B = np.c_[
            data_loaded.survey.locations_a,
            data_loaded.survey.locations_b,
            data_loaded.survey.locations_m,
            data_loaded.survey.locations_n,
        ]

        passed = np.all(np.isclose(A, B))
        self.assertTrue(passed, True)

        # Test for dipole-dipole
        write_dcipoctree_ubc(
            filename, dpdp_data, "volt", "survey", format_type="general"
        )
        data_loaded = read_dcipoctree_ubc(filename, "volt")
        os.remove(filename)

        A = np.c_[
            self.dpdp_survey.locations_a,
            self.dpdp_survey.locations_b,
            self.dpdp_survey.locations_m,
            self.dpdp_survey.locations_n,
        ]

        B = np.c_[
            data_loaded.survey.locations_a,
            data_loaded.survey.locations_b,
            data_loaded.survey.locations_m,
            data_loaded.survey.locations_n,
        ]

        passed = np.all(np.isclose(A, B))
        self.assertTrue(passed, True)

        print("SURVEY FILE IO FOR DCIP3D PASSED")

    def test_io_dpred(self):

        pp_data = Data(survey=self.pp_survey, dobs=self.dobs)
        dpdp_data = Data(survey=self.dpdp_survey, dobs=self.dobs)

        filename = "dpred.dc"

        # Test for pole-pole
        write_dcip3d_ubc(
            filename, pp_data, "secondary_potential", "dpred", format_type="general"
        )
        data_loaded = read_dcip3d_ubc(filename, "secondary_potential")
        os.remove(filename)

        A = np.c_[
            self.pp_survey.locations_a,
            self.pp_survey.locations_b,
            self.pp_survey.locations_m,
            self.pp_survey.locations_n,
            self.dobs,
        ]

        B = np.c_[
            data_loaded.survey.locations_a,
            data_loaded.survey.locations_b,
            data_loaded.survey.locations_m,
            data_loaded.survey.locations_n,
            data_loaded.dobs,
        ]

        passed = np.all(np.isclose(A, B))
        self.assertTrue(passed, True)

        # Test for dipole-dipole
        write_dcipoctree_ubc(
            filename, dpdp_data, "volt", "dpred", format_type="general"
        )
        data_loaded = read_dcipoctree_ubc(filename, "volt")
        os.remove(filename)

        A = np.c_[
            self.dpdp_survey.locations_a,
            self.dpdp_survey.locations_b,
            self.dpdp_survey.locations_m,
            self.dpdp_survey.locations_n,
            self.dobs,
        ]

        B = np.c_[
            data_loaded.survey.locations_a,
            data_loaded.survey.locations_b,
            data_loaded.survey.locations_m,
            data_loaded.survey.locations_n,
            data_loaded.dobs,
        ]

        passed = np.all(np.isclose(A, B))
        self.assertTrue(passed, True)

        print("PREDICTED DATA FILE IO FOR DCIP3D PASSED")

    def test_io_dobs(self):

        pp_data = Data(
            survey=self.pp_survey, dobs=self.dobs, standard_deviation=self.std
        )
        dpdp_data = Data(
            survey=self.dpdp_survey, dobs=self.dobs, standard_deviation=self.std
        )

        filename = "dobs.dc"

        # Test for pole-pole
        write_dcip3d_ubc(
            filename, pp_data, "secondary_potential", "dobs", format_type="general"
        )
        data_loaded = read_dcip3d_ubc(filename, "secondary_potential")
        os.remove(filename)

        A = np.c_[
            self.pp_survey.locations_a,
            self.pp_survey.locations_b,
            self.pp_survey.locations_m,
            self.pp_survey.locations_n,
            self.dobs,
            self.std,
        ]

        B = np.c_[
            data_loaded.survey.locations_a,
            data_loaded.survey.locations_b,
            data_loaded.survey.locations_m,
            data_loaded.survey.locations_n,
            data_loaded.dobs,
            data_loaded.standard_deviation,
        ]

        passed = np.all(np.isclose(A, B))
        self.assertTrue(passed, True)

        # Test for dipole-dipole
        write_dcipoctree_ubc(filename, dpdp_data, "volt", "dobs", format_type="general")
        data_loaded = read_dcipoctree_ubc(filename, "volt")
        os.remove(filename)

        A = np.c_[
            self.dpdp_survey.locations_a,
            self.dpdp_survey.locations_b,
            self.dpdp_survey.locations_m,
            self.dpdp_survey.locations_n,
            self.dobs,
            self.std,
        ]

        B = np.c_[
            data_loaded.survey.locations_a,
            data_loaded.survey.locations_b,
            data_loaded.survey.locations_m,
            data_loaded.survey.locations_n,
            data_loaded.dobs,
            data_loaded.standard_deviation,
        ]

        passed = np.all(np.isclose(A, B))
        self.assertTrue(passed, True)

        print("OBSERVATIONS FILE IO FOR DCIP3D PASSED")


if __name__ == "__main__":
    unittest.main()
