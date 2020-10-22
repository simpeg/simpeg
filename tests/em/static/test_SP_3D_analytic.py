import numpy as np
import unittest

from discretize import TensorMesh
from discretize.utils import mkvc
from discretize.utils.mesh_utils import closest_points_index

from SimPEG import utils, SolverLU, maps
from SimPEG.electromagnetics.static import spontaneous_potential as sp


class SPProblemAnalyticTests_CellCenters(unittest.TestCase):
    """Validate forward modeling for a point charge in a whole space"""
    def setUp(self):

        # Design Mesh
        dh = 0.5
        nc = 40
        npad = 10
        exp = 1.3
        h = [(dh, npad, -exp), (dh, nc), (dh, npad, exp)]
        mesh = TensorMesh([h,h,h], 'CCC')

        # Background conductivity
        sig = 1e-3
        sig_model = sig*np.ones(mesh.nC)

        # Electrode locations
        theta = np.random.uniform(0, np.pi, 10)
        phi = np.random.uniform(-np.pi, np.pi, 10)

        R = 5
        x = R*np.sin(theta)*np.cos(phi)
        y = R*np.sin(theta)*np.sin(phi)
        z = R*np.cos(theta)
        m_loc = np.c_[x, y, z]
        
        R = 7
        x = R*np.sin(theta)*np.cos(phi)
        y = R*np.sin(theta)*np.sin(phi)
        z = R*np.cos(theta)
        n_loc = np.c_[x, y, z]


        self.dh = dh
        self.mesh = mesh
        self.sig = sig
        self.sig_model = sig_model
        self.m_loc = m_loc
        self.n_loc = n_loc

    def test_SimulationCurrentSourceCellCenters_Pole(self, tolerance=0.05):

        rx = [sp.receivers.Pole(self.m_loc)]
        src = [sp.sources.SpontaneousPotentialSource(receivers_list=rx)]
        survey = sp.survey.Survey(source_list=src)

        # Qs is total charge, qs
        Qs = 4.
        qs = np.zeros(self.mesh.nC)
        k = closest_points_index(self.mesh, np.r_[0, 0, 0])
        qs[k] = Qs/self.dh**3  # total charge divided by volume

        # Numerical solution
        qsMap = maps.IdentityMap(nP=self.mesh.nC)
        sim_qs = sp.simulation.SimulationCurrentSourceCellCenters(
            self.mesh, survey=survey, sigma=self.sig_model, qsMap=qsMap
        )
        dpred_num = sim_qs.dpred(qs)

        # Analytic solution
        xyz = mkvc(self.mesh.grid_cell_centers[k, :])
        C = Qs/(4*np.pi*self.sig)

        r = np.sqrt(
            (self.m_loc[:, 0]-xyz[0])**2 + (self.m_loc[:, 1]-xyz[1])**2 + (self.m_loc[:, 2]-xyz[2])**2
        )

        dpred_anal = C * r**-1

        err = np.abs((dpred_anal - dpred_num)/dpred_anal)
        print('ERROR FOR QS SOURCE WITH POLE RECEIVER')
        print(err)
        print('DATA VALUES')
        print(np.c_[dpred_anal, dpred_num])

        # print(f"DPDP N err: {err}")
        # self.assertLess(err, tolerance)


    def test_SimulationCurrentDensityCellCenters_Pole(self, tolerance=0.05):

        rx = [sp.receivers.Pole(self.m_loc)]
        src = [sp.sources.SpontaneousPotentialSource(receivers_list=rx)]
        survey = sp.survey.Survey(source_list=src)

        Qs = 4.
        
        xc = self.mesh.grid_cell_centers[:,0]
        yc = self.mesh.grid_cell_centers[:,1]
        zc = self.mesh.grid_cell_centers[:,2]
        R = np.sqrt(xc**2 + yc**2 + zc**2)

        jsx = (Qs/(4*np.pi)) * xc/R**3
        jsy = (Qs/(4*np.pi)) * yc/R**3
        jsz = (Qs/(4*np.pi)) * zc/R**3
        js = np.r_[jsx, jsy, jsz]

        # Numerical solution
        jsMap = maps.IdentityMap(nP=3*self.mesh.nC)
        sim_qs = sp.simulation.SimulationCurrentDensityCellCenters(
            self.mesh, survey=survey, sigma=self.sig_model, jsMap=jsMap
        )
        dpred_num = sim_qs.dpred(js)

        # Analytic solution
        C = Qs/(4*np.pi*self.sig)
        r = np.sqrt(np.sum(self.m_loc**2, 1))

        dpred_anal = C * r**-1

        err = np.abs((dpred_anal - dpred_num)/dpred_anal)
        print('ERROR FOR JS SOURCE WITH POLE RECEIVER')
        print(err)

        # print(f"DPDP N err: {err}")
        # self.assertLess(err, tolerance)

    def test_SimulationCurrentSourceCellCenters_Dipole(self, tolerance=0.05):

        rx = [sp.receivers.Dipole(self.m_loc, self.n_loc)]
        src = [sp.sources.SpontaneousPotentialSource(receivers_list=rx)]
        survey = sp.survey.Survey(source_list=src)

        # Qs is total charge, qs
        Qs = 4.
        qs = np.zeros(self.mesh.nC)
        k = closest_points_index(self.mesh, np.r_[0, 0, 0])
        qs[k] = Qs/self.dh**3  # total charge divided by volume

        # Numerical solution
        qsMap = maps.IdentityMap(nP=self.mesh.nC)
        sim_qs = sp.simulation.SimulationCurrentSourceCellCenters(
            self.mesh, survey=survey, sigma=self.sig_model, qsMap=qsMap
        )
        dpred_num = sim_qs.dpred(qs)

        # Analytic solution
        xyz = mkvc(self.mesh.grid_cell_centers[k, :])
        C = Qs/(4*np.pi*self.sig)

        r1 = np.sqrt(
            (self.m_loc[:, 0]-xyz[0])**2 + (self.m_loc[:, 1]-xyz[1])**2 + (self.m_loc[:, 2]-xyz[2])**2
        )
        r2 = np.sqrt(
            (self.n_loc[:, 0]-xyz[0])**2 + (self.n_loc[:, 1]-xyz[1])**2 + (self.n_loc[:, 2]-xyz[2])**2
        )

        dpred_anal = C * (r2**-1 - r1**-1)

        err = np.abs((dpred_anal - dpred_num)/dpred_anal)
        print('ERROR FOR QS SOURCE WITH DIPOLE RECEIVER')
        print(err)

        # print(f"DPDP N err: {err}")
        # self.assertLess(err, tolerance)


    def test_SimulationCurrentDensityCellCenters_Dipole(self, tolerance=0.05):

        rx = [sp.receivers.Dipole(self.m_loc, self.n_loc)]
        src = [sp.sources.SpontaneousPotentialSource(receivers_list=rx)]
        survey = sp.survey.Survey(source_list=src)

        Qs = 4.
        
        xc = self.mesh.grid_cell_centers[:,0]
        yc = self.mesh.grid_cell_centers[:,1]
        zc = self.mesh.grid_cell_centers[:,2]
        R = np.sqrt(xc**2 + yc**2 + zc**2)

        jsx = (Qs/(4*np.pi)) * xc/R**3
        jsy = (Qs/(4*np.pi)) * yc/R**3
        jsz = (Qs/(4*np.pi)) * zc/R**3
        js = np.r_[jsx, jsy, jsz]

        # Numerical solution
        jsMap = maps.IdentityMap(nP=3*self.mesh.nC)
        sim_qs = sp.simulation.SimulationCurrentDensityCellCenters(
            self.mesh, survey=survey, sigma=self.sig_model, jsMap=jsMap
        )
        dpred_num = sim_qs.dpred(js)

        # Analytic solution
        C = Qs/(4*np.pi*self.sig)
        r1 = np.sqrt(np.sum(self.m_loc**2, 1))
        r2 = np.sqrt(np.sum(self.n_loc**2, 1))

        dpred_anal = C * (r2**-1 - r1**-1)

        err = np.abs((dpred_anal - dpred_num)/dpred_anal)
        print('ERROR FOR JS SOURCE WITH DIPOLE RECEIVER')
        print(err)

        # print(f"DPDP N err: {err}")
        # self.assertLess(err, tolerance)








if __name__ == "__main__":
    unittest.main()
