# import matplotlib
# matplotlib.use('Agg')
import unittest
import numpy as np

from SimPEG.electromagnetics import resistivity as dc
from SimPEG.electromagnetics.static import utils
from SimPEG import data
from SimPEG.utils.io_utils import io_utils_electromagnetics as io_utils
import shutil
import os


class Test_DCIP_IO(unittest.TestCase):
    def setUp(self):

        self.survey_type = ["pole-pole", "dipole-pole", "pole-dipole", "dipole-dipole"]
        self.topo = 10
        self.num_rx_per_src = 4
        self.station_spacing = 25

        self.dir_path = "./dcip_io_tests"

        os.mkdir(self.dir_path)

    def test_dc2d(self):

        data_type = "volt"
        end_points = np.array([-100, 100])

        # Create sources and data object
        source_list = []
        for stype in self.survey_type:
            source_list = source_list + utils.generate_dcip_sources_line(
                stype,
                data_type,
                "2D",
                end_points,
                self.topo,
                self.num_rx_per_src,
                self.station_spacing,
            )
        survey2D = dc.survey.Survey(source_list)
        dobs = np.random.rand(survey2D.nD)
        dunc = 1e-3 * np.ones(survey2D.nD)
        data2D = data.Data(survey2D, dobs=dobs, standard_deviation=dunc)

        io_utils.write_dcip2d_ubc(
            self.dir_path + "/dc2d_general.txt",
            data2D,
            "volt",
            "dobs",
            "general",
            comment_lines="GENERAL FORMAT",
        )
        io_utils.write_dcip2d_ubc(
            self.dir_path + "/dc2d_surface.txt",
            data2D,
            "volt",
            "dobs",
            "surface",
            comment_lines="SURFACE FORMAT",
        )

        # Read DCIP2D files
        data_general = io_utils.read_dcip2d_ubc(
            self.dir_path + "/dc2d_general.txt", "volt", "general"
        )
        data_surface = io_utils.read_dcip2d_ubc(
            self.dir_path + "/dc2d_surface.txt", "volt", "surface"
        )

        # Compare
        passed = np.all(
            np.isclose(data2D.dobs, data_general.dobs)
            & np.isclose(data2D.dobs, data_surface.dobs)
        )

        self.assertTrue(passed)
        print("READ/WRITE METHODS FOR DC2D DATA PASSED!")

    def test_ip2d(self):

        data_type = "apparent_chargeability"
        end_points = np.array([-100, 100])

        # Create sources and data object
        source_list = []
        for stype in self.survey_type:
            source_list = source_list + utils.generate_dcip_sources_line(
                stype,
                data_type,
                "2D",
                end_points,
                self.topo,
                self.num_rx_per_src,
                self.station_spacing,
            )
        survey2D = dc.survey.Survey(source_list)
        dobs = np.random.rand(survey2D.nD)
        dunc = 1e-3 * np.ones(survey2D.nD)
        data2D = data.Data(survey2D, dobs=dobs, standard_deviation=dunc)

        # Write DCIP2D files
        io_utils.write_dcip2d_ubc(
            self.dir_path + "/ip2d_general.txt",
            data2D,
            "apparent_chargeability",
            "dobs",
            "general",
            comment_lines="GENERAL FORMAT",
        )
        io_utils.write_dcip2d_ubc(
            self.dir_path + "/ip2d_surface.txt",
            data2D,
            "apparent_chargeability",
            "dobs",
            "surface",
            comment_lines="SURFACE FORMAT",
        )

        # Read DCIP2D files
        data_general = io_utils.read_dcip2d_ubc(
            self.dir_path + "/ip2d_general.txt", "apparent_chargeability", "general"
        )
        data_surface = io_utils.read_dcip2d_ubc(
            self.dir_path + "/ip2d_surface.txt", "apparent_chargeability", "surface"
        )

        # Compare
        passed = np.all(
            np.isclose(data2D.dobs, data_general.dobs)
            & np.isclose(data2D.dobs, data_surface.dobs)
        )

        self.assertTrue(passed)
        print("READ/WRITE METHODS FOR IP2D DATA PASSED!")

    def test_dcip3d(self):

        # Survey parameters
        data_type = "volt"
        end_points = np.array([-100, 50, 100, -50])

        # Create sources and data object
        source_list = []
        for stype in self.survey_type:
            source_list = source_list + utils.generate_dcip_sources_line(
                stype,
                data_type,
                "3D",
                end_points,
                self.topo,
                self.num_rx_per_src,
                self.station_spacing,
            )
        survey3D = dc.survey.Survey(source_list)
        dobs = np.random.rand(survey3D.nD)
        dunc = 1e-3 * np.ones(survey3D.nD)
        data3D = data.Data(survey3D, dobs=dobs, standard_deviation=dunc)

        # Write DCIP3D files
        io_utils.write_dcipoctree_ubc(
            self.dir_path + "/dcip3d_general.txt",
            data3D,
            "volt",
            "dobs",
            format_type="general",
            comment_lines="GENERAL FORMAT",
        )
        io_utils.write_dcipoctree_ubc(
            self.dir_path + "/dcip3d_surface.txt",
            data3D,
            "volt",
            "dobs",
            format_type="surface",
            comment_lines="SURFACE FORMAT",
        )

        # Read DCIP3D files
        data_general = io_utils.read_dcipoctree_ubc(
            self.dir_path + "/dcip3d_general.txt", "volt"
        )
        data_surface = io_utils.read_dcipoctree_ubc(
            self.dir_path + "/dcip3d_surface.txt", "volt"
        )

        # Compare
        passed = np.all(
            np.isclose(data3D.dobs, data_general.dobs)
            & np.isclose(data3D.dobs, data_surface.dobs)
        )

        self.assertTrue(passed)
        print("READ/WRITE METHODS FOR DCIP3D DATA PASSED!")

    def tearDown(self):
        # Clean up the working directory
        shutil.rmtree("./dcip_io_tests")


if __name__ == "__main__":
    unittest.main()
