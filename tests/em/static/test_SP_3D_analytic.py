import numpy as np
import unittest

from discretize import TensorMesh
from discretize.utils import mkvc
try:
    from discretize.utils.mesh_utils import closest_points_index
else:
    from discretize.utils.mesh_utils import closestPoints

from SimPEG import utils, SolverLU, maps
from SimPEG.electromagnetics.static import spontaneous_potential as sp


class SPProblemAnalyticTests_CellCenters(unittest.TestCase):
    """Validate forward modeling for a point charge in a whole space"""
    def setUp(self):

        # Design Mesh
        dh = 0.25
        nc = 20
        npad1 = 10
        npad2 = 10
        exp1 = 1.1
        exp2 = 1.5
        hx = [(dh, npad2, -exp2), (dh, npad1, -exp1), (dh, nc), (dh, npad1, exp1), (dh, npad2, exp2)]
        mesh = TensorMesh([hx,hx,hx], 'CCC')
        
        # Background conductivity
        sig = 1e-3
        sig_model = sig*np.ones(mesh.nC)

        # Electrode locations at specified radii from origin
        theta = np.random.uniform(0, np.pi, 10)
        phi = np.random.uniform(-np.pi, np.pi, 10)

        x = np.array([3., 1., 1., -1., -1.])
        y = np.array([0., 3., 0., 3., 0.])
        z = np.array([0., 0., 3., 0., 3.])

        # R = 4
        # x = R*np.sin(theta)*np.cos(phi)
        # y = R*np.sin(theta)*np.sin(phi)
        # z = R*np.cos(theta)
        m_loc = np.c_[x, y, z]
        
        # R = 4
        # x = R*np.sin(theta)*np.cos(phi)
        # y = R*np.sin(theta)*np.sin(phi)
        # z = R*np.cos(theta)

        x = np.array([4., 0., 0., -1., -1.])
        y = np.array([0., 4., 0., 4., 0.])
        z = np.array([0., 0., 4., 0., 4.])
        n_loc = np.c_[x, y, z]

        self.dh = dh
        self.mesh = mesh
        self.sig = sig
        self.sig_model = sig_model
        self.m_loc = m_loc
        self.n_loc = n_loc

    def test_SimulationCurrentSourceCellCenters_Pole(self, tolerance=0.05):
        """
        Simulate response of a +ve electrical charge at (1, 0, 0) and a
        negative charge at (-1, 0, 0) for a pole receiver.
        """

        rx = [sp.receivers.Pole(self.m_loc)]
        src = [sp.sources.SpontaneousPotentialSource(receivers_list=rx)]
        survey = sp.survey.Survey(source_list=src)

        # Qs magnitude of each total charge, qs is the charge divided by smallest cell volume
        Qs = 4.
        qs = np.zeros(self.mesh.nC)
        k1 = closest_points_index(self.mesh, np.r_[1, 0, 0])
        k2 = closest_points_index(self.mesh, np.r_[-1, 0, 0])
        qs[k1] = Qs/self.dh**3  # total charge divided by volume
        qs[k2] = -Qs/self.dh**3  # total charge divided by volume

        # Numerical solution
        qsMap = maps.IdentityMap(nP=self.mesh.nC)
        sim_qs = sp.simulation.SimulationCurrentSourceCellCenters(
            self.mesh, survey=survey, sigma=self.sig_model, qsMap=qsMap
        )
        dpred_num = sim_qs.dpred(qs)

        # Analytic solution
        xyz1 = mkvc(self.mesh.grid_cell_centers[k1, :])
        xyz2 = mkvc(self.mesh.grid_cell_centers[k2, :])
        C = Qs/(4*np.pi*self.sig)

        r1 = np.sqrt(
            (self.m_loc[:, 0]-xyz1[0])**2 +
            (self.m_loc[:, 1]-xyz1[1])**2 +
            (self.m_loc[:, 2]-xyz1[2])**2
        )
        r2 = np.sqrt(
            (self.m_loc[:, 0]-xyz2[0])**2 +
            (self.m_loc[:, 1]-xyz2[1])**2 +
            (self.m_loc[:, 2]-xyz2[2])**2
        )

        dpred_anal = C * (r1**-1 - r2**-1)

        err = np.abs((dpred_anal - dpred_num)/dpred_anal)
        print('\nERROR FOR QS SOURCE WITH POLE RECEIVER')
        print('ANALYTIC | NUMERIC | DIFFERENCE | FRACTIONAL ERROR')
        print(np.c_[dpred_anal, dpred_num, dpred_anal-dpred_num, err])

        tolerance = 0.02
        self.assertLess(np.max(err), tolerance)
        print(f"Maximum fractional error: {np.max(err)} < {tol}")


    def test_SimulationCurrentDensityCellCenters_Pole(self, tolerance=0.05):
        """
        Simulate response of a +ve electrical charge at (1, 0, 0) and a
        negative charge at (-1, 0, 0) for a dipole receiver.
        """

        rx = [sp.receivers.Pole(self.m_loc)]
        src = [sp.sources.SpontaneousPotentialSource(receivers_list=rx)]
        survey = sp.survey.Survey(source_list=src)

        Qs = 4.
        
        x1, y1, z1 = 1, 0, 0
        x2, y2, z2 = -1, 0, 0
        xc = self.mesh.grid_cell_centers[:,0]
        yc = self.mesh.grid_cell_centers[:,1]
        zc = self.mesh.grid_cell_centers[:,2]
        R1 = np.sqrt((xc-x1)**2 + (yc-y1)**2 + (zc-z1)**2)
        R2 = np.sqrt((xc-x2)**2 + (yc-y2)**2 + (zc-z2)**2)

        jsx1 = (Qs/(4*np.pi)) * (xc-x1)/R1**3
        jsy1 = (Qs/(4*np.pi)) * (yc-y1)/R1**3
        jsz1 = (Qs/(4*np.pi)) * (zc-z1)/R1**3

        jsx2 = -(Qs/(4*np.pi)) * (xc-x2)/R2**3
        jsy2 = -(Qs/(4*np.pi)) * (yc-y2)/R2**3
        jsz2 = -(Qs/(4*np.pi)) * (zc-z2)/R2**3
        
        js = np.r_[jsx1, jsy1, jsz1] + np.r_[jsx2, jsy2, jsz2]

        # Numerical solution
        jsMap = maps.IdentityMap(nP=3*self.mesh.nC)
        sim_qs = sp.simulation.SimulationCurrentDensityCellCenters(
            self.mesh, survey=survey, sigma=self.sig_model, jsMap=jsMap
        )
        dpred_num = sim_qs.dpred(js)

        # Analytic solution
        C = Qs/(4*np.pi*self.sig)
        r1 = np.sqrt(np.sum((self.m_loc - np.r_[x1, y1, z1])**2, 1))
        r2 = np.sqrt(np.sum((self.m_loc - np.r_[x2, y2, z2])**2, 1))

        dpred_anal = C * (r1**-1 - r2**-1)

        err = np.abs((dpred_anal - dpred_num)/dpred_anal)
        print('\nERROR FOR QS SOURCE WITH POLE RECEIVER')
        print('ANALYTIC | NUMERIC | DIFFERENCE | FRACTIONAL ERROR')
        print(np.c_[dpred_anal, dpred_num, dpred_anal-dpred_num, err])

        tolerance = 0.02
        self.assertLess(np.max(err), tolerance)
        print(f"Maximum fractional error: {np.max(err)} < {tol}")

    def test_SimulationCurrentSourceCellCenters_Dipole(self, tolerance=0.05):

        rx = [sp.receivers.Dipole(self.m_loc, self.n_loc)]
        src = [sp.sources.SpontaneousPotentialSource(receivers_list=rx)]
        survey = sp.survey.Survey(source_list=src)

        # Qs magnitude of each total charge, qs is the charge divided by smallest cell volume
        Qs = 4.
        qs = np.zeros(self.mesh.nC)
        k1 = closest_points_index(self.mesh, np.r_[1, 0, 0])
        k2 = closest_points_index(self.mesh, np.r_[-1, 0, 0])
        qs[k1] = Qs/self.dh**3  # total charge divided by volume
        qs[k2] = -Qs/self.dh**3  # total charge divided by volume

        # Numerical solution
        qsMap = maps.IdentityMap(nP=self.mesh.nC)
        sim_qs = sp.simulation.SimulationCurrentSourceCellCenters(
            self.mesh, survey=survey, sigma=self.sig_model, qsMap=qsMap
        )
        dpred_num = sim_qs.dpred(qs)

        # Analytic solution
        xyz1 = mkvc(self.mesh.grid_cell_centers[k1, :])
        xyz2 = mkvc(self.mesh.grid_cell_centers[k2, :])
        C = Qs/(4*np.pi*self.sig)

        rm1 = np.sqrt(
            (self.m_loc[:, 0]-xyz1[0])**2 +
            (self.m_loc[:, 1]-xyz1[1])**2 +
            (self.m_loc[:, 2]-xyz1[2])**2
        )
        rm2 = np.sqrt(
            (self.m_loc[:, 0]-xyz2[0])**2 +
            (self.m_loc[:, 1]-xyz2[1])**2 +
            (self.m_loc[:, 2]-xyz2[2])**2
        )
        rn1 = np.sqrt(
            (self.n_loc[:, 0]-xyz1[0])**2 +
            (self.n_loc[:, 1]-xyz1[1])**2 +
            (self.n_loc[:, 2]-xyz1[2])**2
        )
        rn2 = np.sqrt(
            (self.n_loc[:, 0]-xyz2[0])**2 +
            (self.n_loc[:, 1]-xyz2[1])**2 +
            (self.n_loc[:, 2]-xyz2[2])**2
        )

        dpred_anal = C * (rm1**-1 - rm2**-1 - rn1**-1 + rn2**-1)

        err = np.abs((dpred_anal - dpred_num)/dpred_anal)
        print('\nERROR FOR QS SOURCE WITH POLE RECEIVER')
        print('ANALYTIC | NUMERIC | DIFFERENCE | FRACTIONAL ERROR')
        print(np.c_[dpred_anal, dpred_num, dpred_anal-dpred_num, err])

        tolerance = 0.02
        self.assertLess(np.max(err), tolerance)
        print(f"Maximum fractional error: {np.max(err)} < {tol}")


    def test_SimulationCurrentDensityCellCenters_Dipole(self, tolerance=0.05):

        rx = [sp.receivers.Dipole(self.m_loc, self.n_loc)]
        src = [sp.sources.SpontaneousPotentialSource(receivers_list=rx)]
        survey = sp.survey.Survey(source_list=src)

        Qs = 4.
        
        x1, y1, z1 = 1, 0, 0
        x2, y2, z2 = -1, 0, 0
        xc = self.mesh.grid_cell_centers[:,0]
        yc = self.mesh.grid_cell_centers[:,1]
        zc = self.mesh.grid_cell_centers[:,2]
        R1 = np.sqrt((xc-x1)**2 + (yc-y1)**2 + (zc-z1)**2)
        R2 = np.sqrt((xc-x2)**2 + (yc-y2)**2 + (zc-z2)**2)

        jsx1 = (Qs/(4*np.pi)) * (xc-x1)/R1**3
        jsy1 = (Qs/(4*np.pi)) * (yc-y1)/R1**3
        jsz1 = (Qs/(4*np.pi)) * (zc-z1)/R1**3

        jsx2 = -(Qs/(4*np.pi)) * (xc-x2)/R2**3
        jsy2 = -(Qs/(4*np.pi)) * (yc-y2)/R2**3
        jsz2 = -(Qs/(4*np.pi)) * (zc-z2)/R2**3
        
        js = np.r_[jsx1, jsy1, jsz1] + np.r_[jsx2, jsy2, jsz2]

        # Numerical solution
        jsMap = maps.IdentityMap(nP=3*self.mesh.nC)
        sim_qs = sp.simulation.SimulationCurrentDensityCellCenters(
            self.mesh, survey=survey, sigma=self.sig_model, jsMap=jsMap
        )
        dpred_num = sim_qs.dpred(js)

        # Analytic solution
        C = Qs/(4*np.pi*self.sig)
        rm1 = np.sqrt(np.sum((self.m_loc - np.r_[x1, y1, z1])**2, 1))
        rm2 = np.sqrt(np.sum((self.m_loc - np.r_[x2, y2, z2])**2, 1))
        rn1 = np.sqrt(np.sum((self.n_loc - np.r_[x1, y1, z1])**2, 1))
        rn2 = np.sqrt(np.sum((self.n_loc - np.r_[x2, y2, z2])**2, 1))

        dpred_anal = C * (rm1**-1 - rm2**-1 - rn1**-1 + rn2**-1)

        err = np.abs((dpred_anal - dpred_num)/dpred_anal)
        print('\nERROR FOR QS SOURCE WITH POLE RECEIVER')
        print('ANALYTIC | NUMERIC | DIFFERENCE | FRACTIONAL ERROR')
        print(np.c_[dpred_anal, dpred_num, dpred_anal-dpred_num, err])

        tolerance = 0.02
        self.assertLess(np.max(err), tolerance)
        print(f"Maximum fractional error: {np.max(err)} < {tol}")



if __name__ == "__main__":
    unittest.main()
