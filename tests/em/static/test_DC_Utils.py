# import matplotlib
# matplotlib.use('Agg')
import pytest
import unittest
import numpy as np
import discretize

from simpeg.electromagnetics import resistivity as dc
from simpeg.electromagnetics.static import utils
from simpeg import maps, mkvc
from simpeg.utils import io_utils
import shutil
import os


class DCUtilsTests_halfspace(unittest.TestCase):
    def setUp(self):
        url = "https://storage.googleapis.com/simpeg/tests/dc_utils/"
        cloudfiles = [
            "mesh3d.msh",
            "2spheres_conmodel.npy",
            "rhoA_GIF_dd.txt",
            "rhoA_GIF_dp.txt",
            "rhoA_GIF_pd.txt",
            "rhoA_GIF_pp.txt",
        ]

        self.basePath = os.path.expanduser("~/Downloads/TestDCUtilsHalfSpace")
        io_utils.download(
            [url + f for f in cloudfiles], folder=self.basePath, overwrite=True
        )

        # Load Mesh
        mesh_file = os.path.sep.join([self.basePath, "mesh3d.msh"])
        mesh = discretize.load_mesh(mesh_file)
        self.mesh = mesh

        # Load Model
        model_file = os.path.sep.join([self.basePath, "2spheres_conmodel.npy"])
        model = np.load(model_file)
        self.model = model

        xmin, xmax = -15.0, 15.0
        ymin, ymax = 0.0, 0.0
        zmin, zmax = -0.25, -0.25
        xyz = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
        self.xyz = xyz
        self.survey_a = 1.0
        self.survey_b = 1.0
        self.survey_n = 10
        self.plotIt = False

    def test_io_rhoa(self):
        # Setup a dipole-dipole Survey

        for survey_type, test_file, rhoa_file in zip(
            ["dipole-dipole", "pole-dipole", "dipole-pole", "pole-pole"],
            [
                "2sph_dipole_dipole.obs",
                "2sph_pole_dipole.obs",
                "2sph_dipole_pole.obs",
                "2sph_pole_pole.obs",
            ],
            [
                "rhoA_GIF_dd.txt",
                "rhoA_GIF_pd.txt",
                "rhoA_GIF_dp.txt",
                "rhoA_GIF_pp.txt",
            ],
        ):
            print("\n Testing {} ... ".format(survey_type))
            survey = utils.generate_dcip_survey(
                self.xyz,
                survey_type,
                self.survey_a,
                self.survey_b,
                self.survey_n,
                dim=self.mesh.dim,
            )

            # Setup Problem with exponential mapping
            expmap = maps.ExpMap(self.mesh)
            problem = dc.Simulation3DCellCentered(
                self.mesh, sigmaMap=expmap, survey=survey, bc_type="Neumann"
            )

            # Create synthetic data
            dobs = problem.make_synthetic_data(
                self.model, relative_error=0.0, random_seed=40
            )
            dobs.noise_floor = 1e-5

            # Testing IO
            surveyfile = os.path.sep.join([self.basePath, test_file])
            io_utils.write_dcip3d_ubc(
                surveyfile, dobs, "volt", "dobs", format_type="GENERAL"
            )
            data2 = io_utils.read_dcip3d_ubc(surveyfile, "volt")
            self.assertTrue(np.allclose(mkvc(data2), mkvc(dobs)))

            if self.plotIt:
                import matplotlib.pyplot as plt

                # Test Pseudosections plotting
                fig, ax = plt.subplots(1, 1, figsize=(15, 3))
                ax = utils.plot_pseudoSection(
                    survey,
                    ax,
                    survey_type=survey_type,
                    scale="log",
                    clim=None,
                    data_type="appResistivity",
                    pcolor_opts={"cmap": "viridis"},
                    data_location=True,
                )
                plt.show()

            # Test the utils functions electrode_separations,
            # pseudo_locations, geometric_factor,
            # apparent_resistivity all at once
            rhoapp = utils.apparent_resistivity_from_voltage(
                dobs.survey, dobs.dobs, space_type="half-space", eps=0.0
            )

            rhoA_GIF_file = os.path.sep.join([self.basePath, rhoa_file])
            rhoA_GIF = np.loadtxt(rhoA_GIF_file)
            passed = np.allclose(rhoapp, rhoA_GIF)
            self.assertTrue(passed)
            print(f"   ... ok ({survey_type})\n")

    def tearDown(self):
        # Clean up the working directory
        shutil.rmtree(self.basePath)


class DCUtilsTests_fullspace(unittest.TestCase):
    def setUp(self):
        url = "https://storage.googleapis.com/simpeg/tests/dc_utils/"
        cloudfiles = [
            "dPred_fullspace.txt",
            "AB_GIF_fullspace.txt",
            "MN_GIF_fullspace.txt",
            "AM_GIF_fullspace.txt",
            "AN_GIF_fullspace.txt",
            "BM_GIF_fullspace.txt",
            "BN_GIF_fullspace.txt",
            "RhoApp_GIF_fullspace.txt",
        ]

        self.basePath = os.path.expanduser("~/Downloads/TestDCUtilsFullSpace")
        self.files = io_utils.download(
            [url + f for f in cloudfiles], folder=self.basePath, overwrite=True
        )

        survey_file = os.path.sep.join([self.basePath, "dPred_fullspace.txt"])
        data = io_utils.read_dcip3d_ubc(survey_file, "volt")
        self.survey = data.survey
        self.data = data

    def test_ElecSep(self):
        # Compute dipoles distances from survey
        elecSepDict = utils.electrode_separations(self.survey)

        AM = elecSepDict["AM"]
        BM = elecSepDict["BM"]
        AN = elecSepDict["AN"]
        BN = elecSepDict["BN"]
        MN = elecSepDict["MN"]
        AB = elecSepDict["AB"]

        # Load benchmarks files from UBC-GIF codes
        AB_file = os.path.sep.join([self.basePath, "AB_GIF_fullspace.txt"])
        MN_file = os.path.sep.join([self.basePath, "MN_GIF_fullspace.txt"])
        AM_file = os.path.sep.join([self.basePath, "AM_GIF_fullspace.txt"])
        AN_file = os.path.sep.join([self.basePath, "AN_GIF_fullspace.txt"])
        BM_file = os.path.sep.join([self.basePath, "BM_GIF_fullspace.txt"])
        BN_file = os.path.sep.join([self.basePath, "BN_GIF_fullspace.txt"])

        AB_GIF = np.loadtxt(AB_file)
        MN_GIF = np.loadtxt(MN_file)
        AM_GIF = np.loadtxt(AM_file)
        AN_GIF = np.loadtxt(AN_file)
        BM_GIF = np.loadtxt(BM_file)
        BN_GIF = np.loadtxt(BN_file)

        # Assert agreements between the two codes
        test_AB = np.allclose(AB_GIF, AB)
        test_MN = np.allclose(MN_GIF, MN)
        test_AM = np.allclose(AM_GIF, AM)
        test_AN = np.allclose(AN_GIF, AN)
        test_BM = np.allclose(BM_GIF, BM)
        test_BN = np.allclose(BN_GIF, BN)

        passed = np.all([test_AB, test_MN, test_AM, test_AN, test_BM, test_BN])

        self.assertTrue(passed)

    def test_apparent_resistivity_from_voltage(self):
        # Compute apparent resistivity from survey
        rhoapp = utils.apparent_resistivity_from_voltage(
            self.survey, self.data.dobs, space_type="whole-space", eps=1e-16
        )

        # Load benchmarks files from UBC-GIF codes
        rhoappfile = os.path.sep.join([self.basePath, "RhoApp_GIF_fullspace.txt"])
        rhogif = np.loadtxt(rhoappfile)
        # remove value with almost null geometric factor
        idx = rhoapp < 1e8
        # Assert agreements between the two codes
        np.testing.assert_allclose(rhoapp[idx], rhogif[idx], rtol=1e-05, atol=1e-08)

    def tearDown(self):
        # Clean up the working directory
        shutil.rmtree(self.basePath)


class DCUtilsTests_survey_from_ABMN(unittest.TestCase):
    def setUp(self):
        # Define the parameters for each survey line
        survey_type = ["dipole-dipole", "pole-pole", "pole-dipole", "dipole-pole"]
        data_type = "volt"
        dimension_type = "3D"
        end_locations = [
            np.r_[-1000.0, 1000.0, 0.0, 0.0],
            np.r_[0.0, 0.0, -1000.0, 1000.0],
        ]
        station_separation = 200.0
        num_rx_per_src = 5

        # The source lists for each line can be appended to create the source
        # list for the whole survey.
        source_list = []
        lineID = []
        for end_location in end_locations:
            for survey_type_i in survey_type:
                source_list += utils.generate_dcip_sources_line(
                    survey_type_i,
                    data_type,
                    dimension_type,
                    end_location,
                    0.0,
                    num_rx_per_src,
                    station_separation,
                )

        # Define the survey
        self.survey = dc.survey.Survey(source_list)

        lineID = np.ones(len(self.survey.locations_a))
        lineID[int(len(self.survey.locations_a) / 2) :] = 2
        self.lineID = lineID

    def test_generate_survey_from_abmn_locations(self):
        survey_new, sorting_index = utils.generate_survey_from_abmn_locations(
            locations_a=self.survey.locations_a,
            locations_b=self.survey.locations_b,
            locations_m=self.survey.locations_m,
            locations_n=self.survey.locations_n,
            data_type="volt",
            output_sorting=True,
        )

        A = np.c_[
            self.survey.locations_a[sorting_index, :],
            self.survey.locations_b[sorting_index, :],
            self.survey.locations_m[sorting_index, :],
            self.survey.locations_n[sorting_index, :],
        ]

        B = np.c_[
            survey_new.locations_a,
            survey_new.locations_b,
            survey_new.locations_m,
            survey_new.locations_n,
        ]

        passed = np.allclose(A, B)
        self.assertTrue(passed)

    def test_get_source_locations(self):
        # Sources have pole and dipole which impacts unique return
        is_rx = np.all(
            np.isclose(self.survey.locations_m, self.survey.locations_n), axis=1
        )
        is_rx = np.array(is_rx, dtype=float)

        _, idx = np.unique(
            np.c_[self.survey.locations_a, self.survey.locations_b, is_rx, self.lineID],
            axis=0,
            return_index=True,
        )
        src_locations = np.c_[
            self.survey.locations_a[np.sort(idx), :],
            self.survey.locations_b[np.sort(idx), :],
        ]

        src_locations_new = self.survey.source_locations
        has_nan = np.any(np.isnan(src_locations_new[1]), axis=1)
        src_locations_new[1][has_nan, :] = src_locations_new[0][has_nan, :]
        src_locations_new = np.hstack(src_locations_new)

        passed = np.allclose(src_locations, src_locations_new)
        self.assertTrue(passed)

    def test_convert_to_2d(self):
        # 3D survey has two lines of data, one E-W and the other N-S.
        survey_2d_list, IND = utils.convert_survey_3d_to_2d_lines(
            self.survey, self.lineID, data_type="volt", output_indexing=True
        )

        # First, check that coordinates remain the same even after the transformation for the first line
        IND = IND[0]
        survey_2d = survey_2d_list[0]

        ds = np.c_[-1000.0, 0.0, 0.0]

        loc3d = (
            np.r_[
                self.survey.locations_a[IND, :],
                self.survey.locations_b[IND, :],
                self.survey.locations_m[IND, :],
                self.survey.locations_n[IND, :],
            ]
            - ds
        )

        loc2d = np.r_[
            survey_2d.locations_a,
            survey_2d.locations_b,
            survey_2d.locations_m,
            survey_2d.locations_n,
        ]

        # Coordinates should be roughly the same
        passed = np.allclose(loc3d[:, 0::2], loc2d)
        self.assertTrue(passed)

        # Check that the first x-coordinate for electrode A is zero for both surveys
        for survey in survey_2d_list:
            self.assertEqual(survey.locations_a[0, 0], 0)


class TestConvertTo2DInvalidInputs:
    """
    Test convert_survey_3d_to_2d_lines after passing invalid inputs.
    """

    @pytest.fixture
    def survey_3d(self):
        """Sample 3D DC survey."""
        receiver = dc.receivers.Dipole(
            locations_m=np.array([[-100, 0, 0]]),
            locations_n=np.array([[100, 0, 0]]),
            data_type="volt",
        )
        source = dc.sources.Dipole(
            receiver_list=[receiver],
            location_a=np.array([-50, 0, 0]),
            location_b=np.array([50, 0, 0]),
        )
        survey = dc.Survey(source_list=[source])
        return survey

    @pytest.fixture
    def survey_2d(self):
        """Sample 2D DC survey."""
        receiver = dc.receivers.Dipole(
            locations_m=np.array([[-100, 0]]),
            locations_n=np.array([[100, 0]]),
            data_type="volt",
        )
        source = dc.sources.Dipole(
            receiver_list=[receiver],
            location_a=np.array([-50, 0]),
            location_b=np.array([50, 0]),
        )
        survey = dc.Survey(source_list=[source])
        return survey

    def test_invalid_survey(self, survey_2d):
        """
        Test if error is raised when passing an invalid survey (2D survey)
        """
        line_ids = np.ones(survey_2d.nD)
        with pytest.raises(ValueError, match="Invalid 2D 'survey'"):
            utils.convert_survey_3d_to_2d_lines(survey_2d, line_ids)

    def test_invalid_line_ids_wrong_dims(self, survey_3d):
        """
        Test if error is raised after invalid line_ids with wrong dimensions.
        """
        line_ids = np.atleast_2d(np.ones(survey_3d.nD))
        msg = "Invalid 'lineID' array with '2' dimensions. "
        with pytest.raises(ValueError, match=msg):
            utils.convert_survey_3d_to_2d_lines(survey_3d, line_ids)

    def test_invalid_line_ids_wrong_size(self, survey_3d):
        """
        Test if error is raised after an invalid line_ids with wrong size.
        """
        size = survey_3d.nD - 1
        line_ids = np.ones(size)
        msg = f"Invalid 'lineID' array with '{size}' elements. "
        with pytest.raises(ValueError, match=msg):
            utils.convert_survey_3d_to_2d_lines(survey_3d, line_ids)


if __name__ == "__main__":
    unittest.main()
