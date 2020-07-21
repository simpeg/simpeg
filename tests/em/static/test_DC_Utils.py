from __future__ import print_function

# import matplotlib
# matplotlib.use('Agg')
import unittest
import numpy as np
import discretize

from SimPEG.electromagnetics import resistivity as dc
from SimPEG.electromagnetics.static import utils
from SimPEG import maps, mkvc
from SimPEG.utils import io_utils
import shutil
import os

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


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
            survey = utils.gen_DCIPsurvey(
                self.xyz,
                survey_type=survey_type,
                dim=self.mesh.dim,
                a=self.survey_a,
                b=self.survey_b,
                n=self.survey_n,
            )

            self.assertEqual(survey_type, survey.survey_type)

            # Setup Problem with exponential mapping
            expmap = maps.ExpMap(self.mesh)
            problem = dc.Simulation3DCellCentered(
                self.mesh, sigmaMap=expmap, survey=survey, bc_type="Neumann"
            )
            problem.solver = Solver

            # Create synthetic data
            dobs = problem.make_synthetic_data(self.model, relative_error=0.0)
            dobs.eps = 1e-5

            # Testing IO
            surveyfile = os.path.sep.join([self.basePath, test_file])
            utils.writeUBC_DCobs(
                surveyfile, dobs, survey_type=survey_type, dim=3, format_type="GENERAL"
            )
            data2 = utils.readUBC_DC3Dobs(surveyfile)
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
                    pcolorOpts={"cmap": "viridis"},
                    data_location=True,
                )
                plt.show()

            # Test the utils functions electrode_separations,
            # source_receiver_midpoints, geometric_factor,
            # apparent_resistivity all at once
            rhoapp = utils.apparent_resistivity(
                dobs, survey_type=survey_type, space_type="half-space", eps=0.0
            )

            rhoA_GIF_file = os.path.sep.join([self.basePath, rhoa_file])
            rhoA_GIF = np.loadtxt(rhoA_GIF_file)
            passed = np.allclose(rhoapp, rhoA_GIF)
            self.assertTrue(passed)
            print("   ... ok \n".format(survey_type))

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
        data = utils.readUBC_DC3Dobs(survey_file)
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

    def test_apparent_resistivity(self):

        # Compute apparent resistivity from survey
        rhoapp = utils.apparent_resistivity(
            self.data, survey_type="dipole-dipole", space_type="whole-space", eps=1e-16
        )

        # Load benchmarks files from UBC-GIF codes
        rhoappfile = os.path.sep.join([self.basePath, "RhoApp_GIF_fullspace.txt"])
        rhogif = np.loadtxt(rhoappfile)
        # remove value with almost null geometric factor
        idx = rhoapp < 1e8
        # Assert agreements between the two codes
        passed = np.allclose(rhoapp[idx], rhogif[idx])
        self.assertTrue(passed)

    def tearDown(self):
        # Clean up the working directory
        shutil.rmtree(self.basePath)


if __name__ == "__main__":
    unittest.main()
