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
        npad = 7
        hx = [(cs, npad, -1.3), (cs, 21), (cs, npad, 1.3)]
        hy = [(cs, npad, -1.3), (cs, 21), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, 20)]
        mesh = Mesh.TensorMesh([hx, hy, hz], x0="CCN")
        sigma = np.ones(mesh.nC)*1e-2

        x = mesh.vectorCCx[(mesh.vectorCCx > -155.) & (mesh.vectorCCx < 155.)]
        y = mesh.vectorCCy[(mesh.vectorCCy > -155.) & (mesh.vectorCCy < 155.)]

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
        data_ana = phiA-phiB

        rx = DC.Rx.Dipole(M, N)
        src = DC.Src.Dipole([rx], Aloc, Bloc)
        survey = DC.Survey([src])

        self.survey = survey
        self.mesh = mesh
        self.sigma = sigma
        self.data_ana = data_ana

    def test_Problem3D_N(self, tolerance=0.2):
        problem = DC.Problem3D_N(self.mesh, sigma=self.sigma)
        problem.Solver = Solver
        problem.pair(self.survey)
        data = self.survey.dpred()
        err = (
            np.linalg.norm(data - self.data_ana) /
            np.linalg.norm(self.data_ana)
        )
        if err < 0.2:
            print(err)
            passed = True
            print(">> DC analytic test for Problem3D_N is passed")
        else:
            print(err)
            passed = False
            print(">> DC analytic test for Problem3D_N is failed")
        self.assertTrue(passed)

    def test_Problem3D_CC_Mixed(self, tolerance=0.2):
        problem = DC.Problem3D_CC(
            self.mesh, sigma=self.sigma, bc_type='Mixed'
        )
        problem.Solver = Solver
        problem.pair(self.survey)
        data = self.survey.dpred()
        err = (
            np.linalg.norm(data - self.data_ana) /
            np.linalg.norm(self.data_ana)
        )
        if err < tolerance:
            print(err)
            passed = True
            print(">> DC analytic test for Problem3D_CC is passed")
        else:
            print(err)
            passed = False
            print(">> DC analytic test for Problem3D_CC is failed")
        self.assertTrue(passed)

    def test_Problem3D_CC_Neumann(self, tolerance=0.2):
        problem = DC.Problem3D_CC(
            self.mesh, sigma=self.sigma, bc_type='Neumann'
            )
        problem.Solver = Solver
        problem.pair(self.survey)
        data = self.survey.dpred()
        err = (
            np.linalg.norm(data - self.data_ana) /
            np.linalg.norm(self.data_ana)
        )
        if err < tolerance:
            print(err)
            passed = True
            print(">> DC analytic test for Problem3D_CC is passed")
        else:
            print(err)
            passed = False
            print(">> DC analytic test for Problem3D_CC is failed")
        self.assertTrue(passed)


# This is for testing Dirichlet B.C.
# for wholepsace Earth.
class DCProblemAnalyticTests_Dirichlet(unittest.TestCase):

    def setUp(self):

        cs = 25.
        hx = [(cs, 7, -1.3), (cs, 21), (cs, 7, 1.3)]
        hy = [(cs, 7, -1.3), (cs, 21), (cs, 7, 1.3)]
        hz = [(cs, 7, -1.3), (cs, 20), (cs, 7, -1.3)]
        mesh = Mesh.TensorMesh([hx, hy, hz], x0="CCC")
        sigma = np.ones(mesh.nC)*1e-2

        x = mesh.vectorCCx[(mesh.vectorCCx > -155.) & (mesh.vectorCCx < 155.)]
        y = mesh.vectorCCy[(mesh.vectorCCy > -155.) & (mesh.vectorCCy < 155.)]

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
        data_ana = phiA-phiB

        rx = DC.Rx.Dipole(M, N)
        src = DC.Src.Dipole([rx], Aloc, Bloc)
        survey = DC.Survey([src])

        self.survey = survey
        self.mesh = mesh
        self.sigma = sigma
        self.data_ana = data_ana

    def test_Problem3D_CC_Dirichlet(self, tolerance=0.2):
        problem = DC.Problem3D_CC(
            self.mesh, sigma=self.sigma, bc_type='Dirichlet'
        )

        problem.Solver = Solver
        problem.pair(self.survey)
        data = self.survey.dpred()
        err = (
            np.linalg.norm(data - self.data_ana) /
            np.linalg.norm(self.data_ana)
        )
        if err < tolerance:
            print(err)
            passed = True
            print(">> DC analytic test for Problem3D_CC_Dirchlet is passed")
        else:
            print(err)
            passed = False
            print(">> DC analytic test for Problem3D_CC_Dirchlet is failed")
        self.assertTrue(passed)


# This is for Pole-Pole case
class DCProblemAnalyticTests_Mixed(unittest.TestCase):

    def setUp(self):

        cs = 25.
        hx = [(cs, 7, -1.5), (cs, 21), (cs, 7, 1.5)]
        hy = [(cs, 7, -1.5), (cs, 21), (cs, 7, 1.5)]
        hz = [(cs, 7, -1.5), (cs, 20)]
        mesh = Mesh.TensorMesh([hx, hy, hz], x0="CCN")
        sigma = np.ones(mesh.nC)*1e-2

        x = mesh.vectorCCx[(mesh.vectorCCx > -155.) & (mesh.vectorCCx < 155.)]
        y = mesh.vectorCCy[(mesh.vectorCCy > -155.) & (mesh.vectorCCy < 155.)]

        Aloc = np.r_[-200., 0., 0.]

        M = Utils.ndgrid(x, y, np.r_[0.])
        phiA = EM.Analytics.DCAnalytic_Pole_Pole(
            Aloc, M, 1e-2, earth_type="halfspace"
        )
        data_ana = phiA

        rx = DC.Rx.Pole(M)
        src = DC.Src.Pole([rx], Aloc)
        survey = DC.Survey([src])

        self.survey = survey
        self.mesh = mesh
        self.sigma = sigma
        self.data_ana = data_ana

    def test_Problem3D_CC_Mixed(self, tolerance=0.2):
        problem = DC.Problem3D_CC(self.mesh, sigma=self.sigma, bc_type='Mixed')
        problem.Solver = Solver
        problem.pair(self.survey)
        data = self.survey.dpred()
        err = (
            np.linalg.norm(data - self.data_ana) /
            np.linalg.norm(self.data_ana)
        )
        if err < tolerance:
            print(err)
            passed = True
            print(">> DC analytic test for Problem3D_CC_Mixed is passed")
        else:
            print(err)
            passed = False
            print(">> DC analytic test for Problem3D_CC_Mixed is failed")
        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
