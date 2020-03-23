
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG import maps
import numpy as np
from pymatsolver import Pardiso
import os
my_dir = os.path.dirname(__file__)
import unittest

class DCMiniaturizeTest(unittest.TestCase):
    """
    The miniatuize option will have the most effect when
    the number of sources is greater that the number of unique source electrodes
    """
    def setUp(self):
        IO = dc.IO()
        ABMN = np.loadtxt(my_dir + '/resources/mixed_survey.loc')
        A = ABMN[:, :2]
        B = ABMN[:, 2:4]
        M = ABMN[:, 4:6]
        N = ABMN[:, 6:8]

        survey = IO.from_ambn_locations_to_survey(A, B, M, N,
            survey_type='dipole-dipole')

        # add some other receivers and sources to the mix
        electrode_locations = np.unique(np.r_[A, B, M, N], axis=0)

        rx_p = dc.receivers.Pole(electrode_locations[[10, 20, 30, 40]])
        rx_d = dc.receivers.Dipole(electrode_locations[[10, 20, 30, 40]], electrode_locations[[15, 25, 35, 45]])

        tx_pd = dc.sources.Pole([rx_d], electrode_locations[0])
        tx_pp = dc.sources.Pole([rx_p], electrode_locations[2])
        tx_dp = dc.sources.Dipole([rx_p], electrode_locations[0], electrode_locations[5])

        source_list = survey.source_list
        source_list.append(tx_pd)
        source_list.append(tx_pp)
        source_list.append(tx_dp)

        survey = dc.Survey(source_list)
        self.survey = survey

        mesh, inds = IO.set_mesh(dx=10)

        self.sim1 = dc.Simulation2DNodal(
            survey=survey, mesh=mesh, solver=Pardiso,
            storeJ=False, sigmaMap=maps.IdentityMap(mesh), miniaturize=False)

        self.sim2 = dc.Simulation2DNodal(
            survey=survey, mesh=mesh, solver=Pardiso,
            storeJ=False, sigmaMap=maps.IdentityMap(mesh), miniaturize=True)

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

if __name__ == '__main__':
    unittest.main()
