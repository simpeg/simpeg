from __future__ import print_function
import unittest
from SimPEG import Mesh, Utils, EM
import numpy as np
import SimPEG.EM.Static.DC as DC
try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


class DCProblemAnalyticTests(unittest.TestCase):

    def setUp(self):

        cs = 25.
        hx = [(cs, 7, -1.3), (cs, 21), (cs, 7, 1.3)]
        hy = [(cs, 7, -1.3), (cs, 21), (cs, 7, 1.3)]
        hz = [(cs, 7, -1.3), (cs, 20)]
        mesh = Mesh.TensorMesh([hx, hy, hz], x0="CCN")
        sigma = np.ones(mesh.nC)*1e-2

        x = mesh.vectorCCx[(mesh.vectorCCx > -155.) & (mesh.vectorCCx < 155.)]
        y = mesh.vectorCCx[(mesh.vectorCCy > -155.) & (mesh.vectorCCy < 155.)]

        Aloc = np.r_[-200., 0., 0.]
        Bloc = np.r_[200., 0., 0.]
        M = Utils.ndgrid(x-25., y, np.r_[0.])
        N = Utils.ndgrid(x+25., y, np.r_[0.])
        phiA = EM.Analytics.DCAnalytic_Pole_Dipole(
            Aloc, [M, N], 1e-2, earth_type="halfspace"
        )
        phiB = EM.Analytics.DCAnalytic_Pole_Dipole(
            Bloc, [M, N], 1e-2, earth_type="halfspace"
        )
        data_anal = phiA-phiB

        rx = DC.Rx.Dipole(M, N)
        src = DC.Src.Dipole([rx], Aloc, Bloc)
        survey = DC.Survey([src])

        self.survey = survey
        self.mesh = mesh
        self.sigma = sigma
        self.data_anal = data_anal

    def test_Problem3D_N(self):
        problem = DC.Problem3D_N(self.mesh, sigma=self.sigma)
        problem.Solver = Solver
        problem.pair(self.survey)
        data = self.survey.dpred()
        err = (
            np.linalg.norm(data - self.data_anal) /
            np.linalg.norm(self.data_anal)
        )
        if err < 0.2:
            passed = True
            print(">> DC analytic test for Problem3D_N is passed")
        else:
            passed = False
            print(">> DC analytic test for Problem3D_N is failed")
        self.assertTrue(passed)

    def test_Problem3D_CC(self):
        problem = DC.Problem3D_CC(self.mesh, sigma=self.sigma)
        problem.Solver = Solver
        problem.pair(self.survey)
        data = self.survey.dpred()
        err = (
            np.linalg.norm(data - self.data_anal) /
            np.linalg.norm(self.data_anal)
        )
        if err < 0.2:
            passed = True
            print(">> DC analytic test for Problem3D_CC is passed")
        else:
            passed = False
            print(">> DC analytic test for Problem3D_CC is failed")
        self.assertTrue(passed)

class DCProblemAnalyticTests_Dirichlet(unittest.TestCase):

    def setUp(self):

        cs = 25.
        hx = [(cs, 7, -1.3), (cs, 21), (cs, 7, 1.3)]
        hy = [(cs, 7, -1.3), (cs, 21), (cs, 7, 1.3)]
        hz = [(cs, 7, -1.3), (cs, 20), (cs, 7, -1.3)]
        mesh = Mesh.TensorMesh([hx, hy, hz], x0="CCC")
        sigma = np.ones(mesh.nC)*1e-2

        x = mesh.vectorCCx[(mesh.vectorCCx > -155.) & (mesh.vectorCCx < 155.)]
        y = mesh.vectorCCx[(mesh.vectorCCy > -155.) & (mesh.vectorCCy < 155.)]

        Aloc = np.r_[-200., 0., 0.]
        Bloc = np.r_[200., 0., 0.]
        M = Utils.ndgrid(x-25., y, np.r_[0.])
        N = Utils.ndgrid(x+25., y, np.r_[0.])
        phiA = EM.Analytics.DCAnalytic_Pole_Dipole(
            Aloc, [M, N], 1e-2, earth_type="wholespace"
        )
        phiB = EM.Analytics.DCAnalytic_Pole_Dipole(
            Bloc, [M, N], 1e-2, earth_type="wholespace"
        )
        data_anal = phiA-phiB

        rx = DC.Rx.Dipole(M, N)
        src = DC.Src.Dipole([rx], Aloc, Bloc)
        survey = DC.Survey([src])

        self.survey = survey
        self.mesh = mesh
        self.sigma = sigma
        self.data_anal = data_anal


    def test_Problem3D_CC_Dirchlet(self):
        problem = DC.Problem3D_CC(self.mesh, sigma=self.sigma, bc_type = 'Dirchlet')
        problem.Solver = Solver
        problem.pair(self.survey)
        data = self.survey.dpred()
        err = (
            np.linalg.norm(data - self.data_anal) /
            np.linalg.norm(self.data_anal)
        )
        if err < 0.2:
            passed = True
            print(">> DC analytic test for Problem3D_CC_Dirchlet is passed")
        else:
            passed = False
            print(">> DC analytic test for Problem3D_CC_Dirchlet is failed")
        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
