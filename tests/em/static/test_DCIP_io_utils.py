from __future__ import print_function

# import matplotlib
# matplotlib.use('Agg')
import unittest
import numpy as np

from SimPEG.electromagnetics import resistivity as dc
from SimPEG.electromagnetics.static import utils
from SimPEG import maps, mkvc, data
from SimPEG.utils.io_utils import io_utils_electromagnetics as io_utils
import shutil
import os


class Test_DCIP2D_IO(unittest.TestCase):

    def test_dc2d(self):
        
        # Survey parameters   
        survey_type = ['pole-pole', 'dipole-dipole']
        data_type = 'volt'
        end_points = np.array([-100, 100])
        topo = 10
        num_rx_per_src = 4
        station_spacing = 25

        # Create sources and data object
        source_list = []
        for stype in survey_type:
            source_list = source_list + utils.generate_dcip_sources_line(
                stype, data_type, '2D', end_points, topo, num_rx_per_src, station_spacing
            )
        survey2D = dc.survey.Survey(source_list)
        dobs = np.random.rand(survey2D.nD)
        dunc = 1e-3*np.ones(survey2D.nD)
        data2D = data.Data(survey2D, dobs=dobs, standard_deviation=dunc)

        os.mkdir('./dcip2d_tests')

        try:
        # Write DCIP2D files
            io_utils.write_dcip2d_ubc(
                './dcip2d_tests/dcip2d_general.txt', data2D, 'volt', 'dobs', 'general', comment_lines="GENERAL FORMAT"
            )
            io_utils.write_dcip2d_ubc(
                './dcip2d_tests/dcip2d_surface.txt', data2D, 'volt', 'dobs', 'surface', comment_lines="SURFACE FORMAT"
            )
            io_utils.write_dcip2d_ubc(
                './dcip2d_tests/dcip2d_simple.txt', data2D, 'volt', 'dobs', 'simple', comment_lines="SIMPLE FORMAT"
            )

            # Read DCIP2D files
            data_general = io_utils.read_dcip2d_ubc('./dcip2d_tests/dcip2d_general.txt', 'volt', 'general')
            data_surface = io_utils.read_dcip2d_ubc('./dcip2d_tests/dcip2d_surface.txt', 'volt', 'surface')
            data_simple = io_utils.read_dcip2d_ubc('./dcip2d_tests/dcip2d_simple.txt', 'volt', 'simple')

            # Compare
            passed = (np.all(
               np.isclose(data2D.dobs, data_general.dobs) &
               np.isclose(data2D.dobs, data_surface.dobs) &
               np.isclose(data2D.dobs, data_simple.dobs)
            ))

            self.assertTrue(passed)
            shutil.rmtree('./dcip2d_tests')
            print("READ/WRITE METHODS FOR DC2D DATA PASSED!")

        except Exception:
            print(Exception)
            shutil.rmtree('./dcip2d_tests')
            self.assertTrue(False)

    def test_ip2d(self):
        
        # Survey parameters   
        survey_type = ['dipole-pole', 'pole-dipole']
        data_type = 'apparent_chargeability'
        end_points = np.array([-100, 100])
        topo = 10
        num_rx_per_src = 4
        station_spacing = 25

        # Create sources and data object
        source_list = []
        for stype in survey_type:
            source_list = source_list + utils.generate_dcip_sources_line(
                stype, data_type, '2D', end_points, topo, num_rx_per_src, station_spacing
            )
        survey2D = dc.survey.Survey(source_list)
        dobs = np.random.rand(survey2D.nD)
        dunc = 1e-3*np.ones(survey2D.nD)
        data2D = data.Data(survey2D, dobs=dobs, standard_deviation=dunc)

        os.mkdir('./dcip2d_tests')

        try:
            # Write DCIP2D files
            io_utils.write_dcip2d_ubc(
                './dcip2d_tests/dcip2d_general.txt', data2D, 'apparent_chargeability', 'dobs', 'general', comment_lines="GENERAL FORMAT"
            )
            io_utils.write_dcip2d_ubc(
                './dcip2d_tests/dcip2d_surface.txt', data2D, 'apparent_chargeability', 'dobs', 'surface', comment_lines="SURFACE FORMAT"
            )
            io_utils.write_dcip2d_ubc(
                './dcip2d_tests/dcip2d_simple.txt', data2D, 'apparent_chargeability', 'dobs', 'simple', comment_lines="SIMPLE FORMAT"
            )

            # Read DCIP2D files
            data_general = io_utils.read_dcip2d_ubc('./dcip2d_tests/dcip2d_general.txt', 'apparent_chargeability', 'general')
            data_surface = io_utils.read_dcip2d_ubc('./dcip2d_tests/dcip2d_surface.txt', 'apparent_chargeability', 'surface')
            data_simple = io_utils.read_dcip2d_ubc('./dcip2d_tests/dcip2d_simple.txt', 'apparent_chargeability', 'simple')

            # Compare
            passed = (np.all(
               np.isclose(data2D.dobs, data_general.dobs) &
               np.isclose(data2D.dobs, data_surface.dobs) &
               np.isclose(data2D.dobs, data_simple.dobs)
            ))

            self.assertTrue(passed)
            shutil.rmtree('./dcip2d_tests')
            print("READ/WRITE METHODS FOR IP2D DATA PASSED!")

        except Exception:
            print(Exception)
            shutil.rmtree('./dcip2d_tests')
            self.assertTrue(False)


    def test_dcip3d(self):
        
        # Survey parameters   
        survey_type = ['pole-pole', 'dipole-pole', 'pole-dipole', 'dipole-dipole']
        data_type = 'volt'
        end_points = np.array([-100, 50, 100, -50])
        topo = 10
        num_rx_per_src = 4
        station_spacing = 25

        # Create sources and data object
        source_list = []
        for stype in survey_type:
            source_list = source_list + utils.generate_dcip_sources_line(
                stype, data_type, '3D', end_points, topo, num_rx_per_src, station_spacing
            )
        survey3D = dc.survey.Survey(source_list)
        dobs = np.random.rand(survey3D.nD)
        dunc = 1e-3*np.ones(survey3D.nD)
        data3D = data.Data(survey3D, dobs=dobs, standard_deviation=dunc)

        os.mkdir('./dcip3d_tests')

        try:
            # Write DCIP3D files
            io_utils.write_dcipoctree_ubc(
                './dcip3d_tests/dcip3d_general.txt', data3D, 'volt', 'dobs', format_type='general', comment_lines="GENERAL FORMAT"
            )
            io_utils.write_dcipoctree_ubc(
                './dcip3d_tests/dcip3d_surface.txt', data3D, 'volt', 'dobs', format_type='surface', comment_lines="SURFACE FORMAT"
            )

            # Read DCIP3D files
            data_general = io_utils.read_dcipoctree_ubc('./dcip3d_tests/dcip3d_general.txt', 'volt')
            data_surface = io_utils.read_dcipoctree_ubc('./dcip3d_tests/dcip3d_surface.txt', 'volt')

            # Compare
            passed = (np.all(
               np.isclose(data3D.dobs, data_general.dobs) &
               np.isclose(data3D.dobs, data_surface.dobs)
            ))

            self.assertTrue(passed)
            shutil.rmtree('./dcip3d_tests')
            print("READ/WRITE METHODS FOR DCIP3D DATA PASSED!")

        except Exception:
            print(Exception)
            shutil.rmtree('./dcip3d_tests')
            self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
