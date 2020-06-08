from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static.utils.static_utils import gen_DCIPsurvey
from SimPEG import maps
import numpy as np
from pymatsolver import Pardiso
import discretize
import os

my_dir = os.path.dirname(__file__)
import unittest


class DCMini2DTestSurveyTypes(unittest.TestCase):
    """
        Test minaturize logic works for each individual type of survey
    """

    def setUp(self):
        aSpacing = 2.5
        nElecs = 5

        surveySize = nElecs * aSpacing - aSpacing
        cs = surveySize / nElecs / 4

        self.mesh = discretize.TensorMesh(
            [
                [(cs, 10, -1.3), (cs, surveySize / cs), (cs, 10, 1.3)],
                [(cs, 3, -1.3), (cs, 3, 1.3)],
                # [(cs, 5, -1.3), (cs, 10)]
            ],
            "CN",
        )

        survey_end_points = np.array([[-surveySize / 2, 0, 0], [surveySize / 2, 0, 0]])

        self.d_d_survey = gen_DCIPsurvey(
            survey_end_points, "dipole-dipole", aSpacing, aSpacing, nElecs, dim=2
        )
        self.d_p_survey = gen_DCIPsurvey(
            survey_end_points, "dipole-pole", aSpacing, aSpacing, nElecs, dim=2
        )
        self.p_d_survey = gen_DCIPsurvey(
            survey_end_points, "pole-dipole", aSpacing, aSpacing, nElecs, dim=2
        )
        self.p_p_survey = gen_DCIPsurvey(
            survey_end_points, "pole-pole", aSpacing, aSpacing, nElecs, dim=2
        )

    def test_dipole_dipole_mini(self):
        sim1 = dc.Simulation2DNodal(
            mesh=self.mesh, survey=self.d_d_survey, sigmaMap=maps.IdentityMap(self.mesh)
        )

        sim2 = dc.Simulation2DNodal(
            mesh=self.mesh,
            survey=self.d_d_survey,
            sigmaMap=maps.IdentityMap(self.mesh),
            miniaturize=True,
        )

        mSynth = np.ones(self.mesh.nC)

        d1 = sim1.dpred(mSynth)
        d2 = sim2.dpred(mSynth)
        self.assertTrue(np.allclose(d1, d2))

    def test_dipole_pole_mini(self):
        sim1 = dc.Simulation2DNodal(
            mesh=self.mesh, survey=self.d_p_survey, sigmaMap=maps.IdentityMap(self.mesh)
        )

        sim2 = dc.Simulation2DNodal(
            mesh=self.mesh,
            survey=self.d_p_survey,
            sigmaMap=maps.IdentityMap(self.mesh),
            miniaturize=True,
        )

        mSynth = np.ones(self.mesh.nC)

        d1 = sim1.dpred(mSynth)
        d2 = sim2.dpred(mSynth)
        self.assertTrue(np.allclose(d1, d2))

    def test_pole_dipole_mini(self):
        sim1 = dc.Simulation2DNodal(
            mesh=self.mesh, survey=self.p_d_survey, sigmaMap=maps.IdentityMap(self.mesh)
        )

        sim2 = dc.Simulation2DNodal(
            mesh=self.mesh,
            survey=self.p_d_survey,
            sigmaMap=maps.IdentityMap(self.mesh),
            miniaturize=True,
        )

        mSynth = np.ones(self.mesh.nC)

        d1 = sim1.dpred(mSynth)
        d2 = sim2.dpred(mSynth)
        self.assertTrue(np.allclose(d1, d2))

    def test_pole_pole_mini(self):
        sim1 = dc.Simulation2DNodal(
            mesh=self.mesh, survey=self.p_p_survey, sigmaMap=maps.IdentityMap(self.mesh)
        )

        sim2 = dc.Simulation2DNodal(
            mesh=self.mesh,
            survey=self.p_p_survey,
            sigmaMap=maps.IdentityMap(self.mesh),
            miniaturize=True,
        )

        mSynth = np.ones(self.mesh.nC)

        d1 = sim1.dpred(mSynth)
        d2 = sim2.dpred(mSynth)
        self.assertTrue(np.allclose(d1, d2))


class DC2DMiniaturizeTest(unittest.TestCase):
    """
    The miniatuize option will have the most effect when
    the number of sources is greater that the number of unique source electrodes
    """

    def setUp(self):
        IO = dc.IO()
        ABMN = np.loadtxt(my_dir + "/resources/mixed_survey.loc")
        A = ABMN[:100, :2]
        B = ABMN[:100, 2:4]
        M = ABMN[:100, 4:6]
        N = ABMN[:100, 6:8]

        survey = IO.from_ambn_locations_to_survey(
            A, B, M, N, survey_type="dipole-dipole"
        )

        # add some other receivers and sources to the mix
        electrode_locations = np.unique(np.r_[A, B, M, N], axis=0)

        rx_p = dc.receivers.Pole(electrode_locations[[2, 4, 6]])
        rx_d = dc.receivers.Dipole(
            electrode_locations[[2, 4, 6]], electrode_locations[[3, 5, 9]]
        )

        tx_pd = dc.sources.Pole([rx_d], electrode_locations[0])
        tx_pp = dc.sources.Pole([rx_p], electrode_locations[1])
        tx_dp = dc.sources.Dipole(
            [rx_p], electrode_locations[0], electrode_locations[1]
        )

        source_list = survey.source_list
        source_list.append(tx_pd)
        source_list.append(tx_pp)
        source_list.append(tx_dp)

        survey = dc.Survey(source_list)
        self.survey = survey
        # This survey is a mix of d-d, d-p, p-d, and p-p txs and rxs.

        # This mesh is meant only for testing
        mesh, inds = IO.set_mesh(dx=10, dz=40)

        self.sim1 = dc.Simulation2DNodal(
            survey=survey,
            mesh=mesh,
            solver=Pardiso,
            storeJ=False,
            sigmaMap=maps.IdentityMap(mesh),
            miniaturize=False,
        )

        self.sim2 = dc.Simulation2DNodal(
            survey=survey,
            mesh=mesh,
            solver=Pardiso,
            storeJ=False,
            sigmaMap=maps.IdentityMap(mesh),
            miniaturize=True,
        )

        self.model = np.ones(mesh.nC)
        self.f1 = self.sim1.fields(self.model)
        self.f2 = self.sim2.fields(self.model)

    def test_dpred(self):
        d1 = self.sim1.dpred(self.model, f=self.f1)
        d2 = self.sim2.dpred(self.model, f=self.f2)
        self.assertTrue(np.allclose(d1, d2))

    def test_Jvec(self):
        u = np.random.rand(*self.model.shape)
        J1u = self.sim1.Jvec(self.model, u, f=self.f1)
        J2u = self.sim2.Jvec(self.model, u, f=self.f2)
        self.assertTrue(np.allclose(J1u, J2u))

    def test_Jtvec(self):
        v = np.random.rand(self.survey.nD)
        J1tv = self.sim1.Jtvec(self.model, v, f=self.f1)
        J2tv = self.sim2.Jtvec(self.model, v, f=self.f2)
        self.assertTrue(np.allclose(J1tv, J2tv))

    def test_J(self):
        J1 = self.sim1.getJ(self.model, f=self.f1)
        J2 = self.sim2.getJ(self.model, f=self.f2)
        self.assertTrue(np.allclose(J1, J2))


class DC3DMiniaturizeTest(unittest.TestCase):
    """
    The miniatuize option will have the most effect when
    the number of sources is greater that the number of unique source electrodes
    """

    def setUp(self):
        aSpacing = 2.5
        nElecs = 5

        surveySize = nElecs * aSpacing - aSpacing
        cs = surveySize / nElecs / 4

        mesh = discretize.TensorMesh(
            [
                [(cs, 10, -1.3), (cs, surveySize / cs), (cs, 10, 1.3)],
                [(cs, 3, -1.3), (cs, 3, 1.3)],
                # [(cs, 5, -1.3), (cs, 10)]
            ],
            "CN",
        )

        survey_end_points = np.array([[-surveySize / 2, 0, 0], [surveySize / 2, 0, 0]])

        survey = gen_DCIPsurvey(
            survey_end_points, "dipole-dipole", aSpacing, aSpacing, nElecs, dim=2
        )
        A = survey.locations_a
        B = survey.locations_b
        M = survey.locations_m
        N = survey.locations_n
        # add some other receivers and sources to the mix
        electrode_locations = np.unique(np.r_[A, B, M, N], axis=0)

        rx_p = dc.receivers.Pole(electrode_locations[[2]])
        rx_d = dc.receivers.Dipole(electrode_locations[[2]], electrode_locations[[3]])

        tx_pd = dc.sources.Pole([rx_d], electrode_locations[0])
        tx_pp = dc.sources.Pole([rx_p], electrode_locations[0])
        tx_dp = dc.sources.Dipole(
            [rx_p], electrode_locations[0], electrode_locations[1]
        )

        source_list = survey.source_list
        source_list.append(tx_pd)
        source_list.append(tx_pp)
        source_list.append(tx_dp)

        survey = dc.Survey(source_list)
        self.survey = survey
        # This survey is a mix of d-d, d-p, p-d, and p-p txs and rxs.

        self.sim1 = dc.Simulation3DNodal(
            survey=survey,
            mesh=mesh,
            solver=Pardiso,
            storeJ=False,
            sigmaMap=maps.IdentityMap(mesh),
            miniaturize=False,
        )

        self.sim2 = dc.Simulation3DNodal(
            survey=survey,
            mesh=mesh,
            solver=Pardiso,
            storeJ=False,
            sigmaMap=maps.IdentityMap(mesh),
            miniaturize=True,
        )

        self.model = np.ones(mesh.nC)
        self.f1 = self.sim1.fields(self.model)
        self.f2 = self.sim2.fields(self.model)

    def test_dpred(self):
        d1 = self.sim1.dpred(self.model, f=self.f1)
        d2 = self.sim2.dpred(self.model, f=self.f2)
        self.assertTrue(np.allclose(d1, d2))

    def test_Jvec(self):
        u = np.random.rand(*self.model.shape)
        J1u = self.sim1.Jvec(self.model, u, f=self.f1)
        J2u = self.sim2.Jvec(self.model, u, f=self.f2)
        self.assertTrue(np.allclose(J1u, J2u))

    def test_Jtvec(self):
        v = np.random.rand(self.survey.nD)
        J1tv = self.sim1.Jtvec(self.model, v, f=self.f1)
        J2tv = self.sim2.Jtvec(self.model, v, f=self.f2)
        self.assertTrue(np.allclose(J1tv, J2tv))

    def test_J(self):
        J1 = self.sim1.getJ(self.model, f=self.f1)
        J2 = self.sim2.getJ(self.model, f=self.f2)
        self.assertTrue(np.allclose(J1, J2))


if __name__ == "__main__":
    unittest.main()
