from __future__ import print_function
# import matplotlib
# matplotlib.use('Agg')
import unittest
import numpy as np
from SimPEG.EM.Static import DC, Utils as DCUtils
from SimPEG import Mesh, Maps
from SimPEG.Utils import io_utils
import shutil
import os

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


class DCUtilsTests_halfspace(unittest.TestCase):

    def setUp(self):
        url = 'https://storage.googleapis.com/simpeg/tests/dc_utils/'
        cloudfiles = [
            'mesh3d.msh', '2spheres_conmodel.npy',
            'rhoA_GIF_dd.txt', 'rhoA_GIF_dp.txt',
            'rhoA_GIF_pd.txt', 'rhoA_GIF_pp.txt'
        ]

        self.basePath = os.path.expanduser('~/Downloads/TestDCUtilsHalfSpace')
        self.files = io_utils.download(
            [url+f for f in cloudfiles],
            folder=self.basePath,
            overwrite=True
        )

        # Load Mesh
        mesh_file = os.path.sep.join([self.basePath, 'mesh3d.msh'])
        mesh = Mesh.load_mesh(mesh_file)
        self.mesh = mesh

        # Load Model
        model_file = os.path.sep.join([self.basePath, '2spheres_conmodel.npy'])
        model = np.load(model_file)
        self.model = model

        xmin, xmax = -15., 15.
        ymin, ymax = 0., 0.
        zmin, zmax = -0.25, -0.25
        xyz = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
        self.xyz = xyz
        self.survey_a = 1.
        self.survey_b = 1.
        self.survey_n = 10
        self.plotIt = False

    def test_dipole_dipole(self):
        # Setup a dipole-dipole Survey

        survey = DCUtils.gen_DCIPsurvey(
            self.xyz,
            survey_type="dipole-dipole",
            dim=self.mesh.dim,
            a=self.survey_a,
            b=self.survey_b,
            n=self.survey_n
        )

        # Setup Problem with exponential mapping
        expmap = Maps.ExpMap(self.mesh)
        problem = DC.Problem3D_CC(self.mesh, sigmaMap=expmap)
        problem.pair(survey)
        problem.Solver = Solver

        # Create synthetic data
        survey.makeSyntheticData(self.model, std=0., force=True)
        survey.eps = 1e-5

        # Testing IO
        surveyfile = os.path.sep.join(
            [self.basePath, '2sph_dipole_dipole.obs']
        )
        DCUtils.writeUBC_DCobs(
            surveyfile,
            survey,
            dim=3,
            format_type='GENERAL'
        )
        survey = DCUtils.readUBC_DC3Dobs(surveyfile)
        survey = survey['dc_survey']
        DCUtils.writeUBC_DCobs(
            surveyfile,
            survey,
            dim=3,
            format_type='GENERAL'
        )
        survey = DCUtils.readUBC_DC3Dobs(surveyfile)
        survey = survey['dc_survey']

        if self.plotIt:
            import matplotlib.pyplot as plt
            # Test Pseudosections plotting
            fig, ax = plt.subplots(1, 1, figsize=(15, 3))
            ax = DCUtils.plot_pseudoSection(
                survey, ax, survey_type='dipole-dipole',
                scale='log', clim=None,
                data_type='appResistivity',
                pcolorOpts={"cmap": "viridis"},
                data_location=True
            )
            plt.show()

        # Test the utils functions electrode_separations,
        # source_receiver_midpoints, geometric_factor,
        # apparent_resistivity all at once
        rhoapp = DCUtils.apparent_resistivity(
            survey, survey_type='dipole-dipole',
            space_type='half-space', eps=0.
        )

        rhoA_GIF_file = os.path.sep.join([self.basePath, 'rhoA_GIF_dd.txt'])
        rhoA_GIF_dd = np.loadtxt(rhoA_GIF_file)
        passed = np.allclose(rhoapp, rhoA_GIF_dd)
        self.assertTrue(passed)

    def test_pole_dipole(self):
        # Setup a pole-dipole Survey

        survey = DCUtils.gen_DCIPsurvey(
            self.xyz,
            survey_type="pole-dipole",
            dim=self.mesh.dim,
            a=self.survey_a,
            b=self.survey_b,
            n=self.survey_n
        )

        # Setup Problem with exponential mapping
        expmap = Maps.ExpMap(self.mesh)
        problem = DC.Problem3D_CC(self.mesh, sigmaMap=expmap)
        problem.pair(survey)
        problem.Solver = Solver

        # Create synthetic data
        survey.makeSyntheticData(self.model, std=0., force=True)
        survey.eps = 1e-5

        # Testing IO
        surveyfile = os.path.sep.join([self.basePath, '2sph_pole_dipole.obs'])
        DCUtils.writeUBC_DCobs(
            surveyfile,
            survey, survey_type='pole-dipole',
            dim=3, format_type='GENERAL'
        )
        survey = DCUtils.readUBC_DC3Dobs(surveyfile)
        survey = survey['dc_survey']
        DCUtils.writeUBC_DCobs(
            surveyfile,
            survey, survey_type='pole-dipole',
            dim=3, format_type='GENERAL'
        )
        survey = DCUtils.readUBC_DC3Dobs(surveyfile)
        survey = survey['dc_survey']

        if self.plotIt:
            import matplotlib.pyplot as plt
            # Test Pseudosections plotting
            fig, ax = plt.subplots(1, 1, figsize=(15, 3))
            ax = DCUtils.plot_pseudoSection(
                survey, ax, survey_type='pole-dipole',
                scale='log', clim=None,
                data_type='appResistivity',
                pcolorOpts={"cmap": "viridis"},
                data_location=True
            )
            plt.show()

        # Test the utils functions electrode_separations,
        # source_receiver_midpoints, geometric_factor,
        # apparent_resistivity all at once
        rhoapp = DCUtils.apparent_resistivity(
            survey, survey_type='pole-dipole',
            space_type='half-space', eps=0.
        )

        rhoA_GIF_file = os.path.sep.join([self.basePath, 'rhoA_GIF_pd.txt'])
        rhoA_GIF_pd = np.loadtxt(rhoA_GIF_file)
        passed = np.allclose(rhoapp, rhoA_GIF_pd)
        self.assertTrue(passed)

    def test_dipole_pole(self):
        # Setup a dipole-pole Survey

        survey = DCUtils.gen_DCIPsurvey(
            self.xyz,
            survey_type="dipole-pole",
            dim=self.mesh.dim,
            a=self.survey_a,
            b=self.survey_b,
            n=self.survey_n
        )

        # Setup Problem with exponential mapping
        expmap = Maps.ExpMap(self.mesh)
        problem = DC.Problem3D_CC(self.mesh, sigmaMap=expmap)
        problem.pair(survey)
        problem.Solver = Solver

        # Create synthetic data
        survey.makeSyntheticData(self.model, std=0., force=True)
        survey.eps = 1e-5

        # Testing IO
        surveyfile = os.path.sep.join([self.basePath, '2sph_dipole_pole.obs'])
        DCUtils.writeUBC_DCobs(
            surveyfile,
            survey, survey_type='dipole-pole',
            dim=3, format_type='GENERAL'
        )
        survey = DCUtils.readUBC_DC3Dobs(surveyfile)
        survey = survey['dc_survey']
        DCUtils.writeUBC_DCobs(
            surveyfile,
            survey, survey_type='dipole-pole',
            dim=3, format_type='GENERAL'
        )
        survey = DCUtils.readUBC_DC3Dobs(surveyfile)
        survey = survey['dc_survey']

        if self.plotIt:
            import matplotlib.pyplot as plt
            # Test Pseudosections plotting
            fig, ax = plt.subplots(1, 1, figsize=(15, 3))
            ax = DCUtils.plot_pseudoSection(
                survey, ax, survey_type='dipole-pole',
                scale='log', clim=None,
                data_type='appResistivity',
                pcolorOpts={"cmap": "viridis"},
                data_location=True
            )
            plt.show()

        # Test the utils functions electrode_separations,
        # source_receiver_midpoints, geometric_factor,
        # apparent_resistivity all at once
        rhoapp = DCUtils.apparent_resistivity(
            survey, survey_type='dipole-pole',
            space_type='half-space', eps=0.)

        rhoA_GIF_file = os.path.sep.join([self.basePath, 'rhoA_GIF_dp.txt'])
        rhoA_GIF_dp = np.loadtxt(rhoA_GIF_file)
        passed = np.allclose(rhoapp, rhoA_GIF_dp)
        self.assertTrue(passed)

    def test_pole_pole(self):
        # Setup a pole-pole Survey

        survey = DCUtils.gen_DCIPsurvey(
            self.xyz,
            survey_type="pole-pole",
            dim=self.mesh.dim,
            a=self.survey_a,
            b=self.survey_b,
            n=self.survey_n
        )

        # Setup Problem with exponential mapping
        expmap = Maps.ExpMap(self.mesh)
        problem = DC.Problem3D_CC(self.mesh, sigmaMap=expmap)
        problem.pair(survey)
        problem.Solver = Solver

        # Create synthetic data
        survey.makeSyntheticData(self.model, std=0., force=True)
        survey.eps = 1e-5

        # Testing IO
        surveyfile = os.path.sep.join([self.basePath, '2sph_pole_pole.obs'])
        DCUtils.writeUBC_DCobs(
            surveyfile,
            survey, survey_type='pole-pole',
            dim=3, format_type='GENERAL'
        )
        survey = DCUtils.readUBC_DC3Dobs(surveyfile)
        survey = survey['dc_survey']
        DCUtils.writeUBC_DCobs(
            surveyfile,
            survey, survey_type='pole-pole',
            dim=3, format_type='GENERAL'
        )
        survey = DCUtils.readUBC_DC3Dobs(surveyfile)
        survey = survey['dc_survey']

        if self.plotIt:
            import matplotlib.pyplot as plt
            # Test Pseudosections plotting
            fig, ax = plt.subplots(1, 1, figsize=(15, 3))
            ax = DCUtils.plot_pseudoSection(
                survey, ax, survey_type='pole-pole',
                scale='log', clim=None,
                data_type='appResistivity',
                pcolorOpts={"cmap": "viridis"},
                data_location=True
            )
            plt.show()
        # Test the utils functions electrode_separations,
        # source_receiver_midpoints, geometric_factor,
        # apparent_resistivity all at once
        rhoapp = DCUtils.apparent_resistivity(
            survey, survey_type='pole-pole',
            space_type='half-space', eps=0.)

        rhoA_GIF_file = os.path.sep.join([self.basePath, 'rhoA_GIF_pp.txt'])
        rhoA_GIF_pp = np.loadtxt(rhoA_GIF_file)
        passed = np.allclose(rhoapp, rhoA_GIF_pp)
        self.assertTrue(passed)

        # Clean up the working directory
        shutil.rmtree(self.basePath)


class DCUtilsTests_fullspace(unittest.TestCase):

    def setUp(self):

        url = 'https://storage.googleapis.com/simpeg/tests/dc_utils/'
        cloudfiles = [
            'dPred_fullspace.txt', 'AB_GIF_fullspace.txt',
            'MN_GIF_fullspace.txt', 'AM_GIF_fullspace.txt',
            'AN_GIF_fullspace.txt', 'BM_GIF_fullspace.txt',
            'BN_GIF_fullspace.txt', 'RhoApp_GIF_fullspace.txt'
        ]

        self.basePath = os.path.expanduser('~/Downloads/TestDCUtilsFullSpace')
        self.files = io_utils.download(
            [url+f for f in cloudfiles],
            folder=self.basePath,
            overwrite=True
        )

        survey_file = os.path.sep.join([self.basePath, 'dPred_fullspace.txt'])
        DCsurvey = DCUtils.readUBC_DC3Dobs(survey_file)
        DCsurvey = DCsurvey['dc_survey']
        self.survey = DCsurvey

    def test_ElecSep(self):

        # Compute dipoles distances from survey
        elecSepDict = DCUtils.electrode_separations(self.survey)

        AM = elecSepDict['AM']
        BM = elecSepDict['BM']
        AN = elecSepDict['AN']
        BN = elecSepDict['BN']
        MN = elecSepDict['MN']
        AB = elecSepDict['AB']

        # Load benchmarks files from UBC-GIF codes
        AB_file = os.path.sep.join([self.basePath, 'AB_GIF_fullspace.txt'])
        MN_file = os.path.sep.join([self.basePath, 'MN_GIF_fullspace.txt'])
        AM_file = os.path.sep.join([self.basePath, 'AM_GIF_fullspace.txt'])
        AN_file = os.path.sep.join([self.basePath, 'AN_GIF_fullspace.txt'])
        BM_file = os.path.sep.join([self.basePath, 'BM_GIF_fullspace.txt'])
        BN_file = os.path.sep.join([self.basePath, 'BN_GIF_fullspace.txt'])

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
        rhoapp = DCUtils.apparent_resistivity(
            self.survey, dobs=self.survey.dobs,
            survey_type='dipole-dipole',
            space_type='whole-space', eps=1e-16
        )

        # Load benchmarks files from UBC-GIF codes
        rhoappfile = os.path.sep.join(
            [self.basePath, 'RhoApp_GIF_fullspace.txt']
        )
        rhogif = np.loadtxt(rhoappfile)
        # remove value with almost null geometric factor
        idx = rhoapp < 1e8
        # Assert agreements between the two codes
        passed = np.allclose(rhoapp[idx], rhogif[idx])
        self.assertTrue(passed)

        # Clean up the working directory
        shutil.rmtree(self.basePath)


if __name__ == '__main__':
    unittest.main()
