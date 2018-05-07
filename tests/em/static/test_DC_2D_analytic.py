from __future__ import print_function
import unittest
from SimPEG import Mesh, Utils, EM, SolverLU
import numpy as np
import SimPEG.EM.Static.DC as DC
import matplotlib.pyplot as plt


class DCProblemAnalyticTests_PDP(unittest.TestCase):

    def setUp(self):

        cs = 12.5
        hx = [(cs, 7, -1.3), (cs, 61), (cs, 7, 1.3)]
        hy = [(cs, 7, -1.3), (cs, 20)]
        mesh = Mesh.TensorMesh([hx, hy], x0="CN")
        sighalf = 1e-2
        sigma = np.ones(mesh.nC)*sighalf
        x = np.linspace(-135, 250., 20)
        M = Utils.ndgrid(x-12.5, np.r_[0.])
        N = Utils.ndgrid(x+12.5, np.r_[0.])
        A0loc = np.r_[-150, 0.]
        # A1loc = np.r_[-130, 0.]
        rxloc = [np.c_[M, np.zeros(20)], np.c_[N, np.zeros(20)]]
        data_ana = EM.Analytics.DCAnalytic_Pole_Dipole(
            np.r_[A0loc, 0.], rxloc, sighalf, earth_type="halfspace"
        )

        rx = DC.Rx.Dipole_ky(M, N)
        src0 = DC.Src.Pole([rx], A0loc)
        survey = DC.Survey_ky([src0])

        self.survey = survey
        self.mesh = mesh
        self.sigma = sigma
        self.data_ana = data_ana
        self.plotIt = False

        try:
            from pymatsolver import Pardiso
            self.Solver = Pardiso
        except ImportError:
            self.Solver = SolverLU

    def test_Problem2D_N(self, tolerance=0.05):

        problem = DC.Problem2D_N(self.mesh, sigma=self.sigma)
        problem.Solver = self.Solver
        problem.pair(self.survey)
        data = self.survey.dpred()
        err = (
            np.linalg.norm((data-self.data_ana) / self.data_ana)**2 /
            self.data_ana.size
        )
        if err < tolerance:
            passed = True
            print(">> DC analytic test for PDP Problem2D_N is passed")
        else:
            print(err)
            passed = False
            print(">> DC analytic test for PDP Problem2D_N is failed")
        self.assertTrue(passed)

    def test_Problem2D_CC(self, tolerance=0.05):
        problem = DC.Problem2D_CC(self.mesh, sigma=self.sigma)
        problem.Solver = self.Solver
        problem.pair(self.survey)
        data = self.survey.dpred()
        err = (
            np.linalg.norm((data-self.data_ana)/self.data_ana)**2 /
            self.data_ana.size
        )
        if err < tolerance:
            passed = True
            print(">> DC analytic test for PDP Problem2D_CC is passed")
        else:
            print(err)
            passed = False
            print(">> DC analytic test for PDP Problem2D_CC is failed")
        self.assertTrue(passed)


class DCProblemAnalyticTests_DPP(unittest.TestCase):

    def setUp(self):

        cs = 12.5
        hx = [(cs, 7, -1.3), (cs, 61), (cs, 7, 1.3)]
        hy = [(cs, 7, -1.3), (cs, 20)]
        mesh = Mesh.TensorMesh([hx, hy], x0="CN")
        sighalf = 1e-2
        sigma = np.ones(mesh.nC)*sighalf
        x = np.linspace(0, 250., 20)
        M = Utils.ndgrid(x-12.5, np.r_[0.])
        N = Utils.ndgrid(x+12.5, np.r_[0.])
        A0loc = np.r_[-150, 0.]
        A1loc = np.r_[-125, 0.]
        rxloc = np.c_[M, np.zeros(20)]
        data_ana = EM.Analytics.DCAnalytic_Dipole_Pole(
                    [np.r_[A0loc, 0.], np.r_[A1loc, 0.]],
                    rxloc, sighalf, earth_type="halfspace")

        rx = DC.Rx.Pole_ky(M)
        src0 = DC.Src.Dipole([rx], A0loc, A1loc)
        survey = DC.Survey_ky([src0])

        self.survey = survey
        self.mesh = mesh
        self.sigma = sigma
        self.data_ana = data_ana
        self.plotIt = False

        try:
            from pymatsolver import PardisoSolver
            self.Solver = PardisoSolver
        except ImportError:
            self.Solver = SolverLU

    def test_Problem2D_N(self, tolerance=0.05):

        problem = DC.Problem2D_N(self.mesh, sigma=self.sigma)
        problem.Solver = self.Solver
        problem.pair(self.survey)
        data = self.survey.dpred()
        err = (
            np.linalg.norm((data-self.data_ana) / self.data_ana)**2 /
            self.data_ana.size
        )
        if err < tolerance:
            passed = True
            print(">> DC analytic test for DPP Problem2D_N is passed")
            if self.plotIt:
                plt.plot(self.data_ana)
                plt.plot(data, 'k.')
                plt.show()
        else:
            passed = False
            print(">> DC analytic test for DPP Problem2D_N is failed")
            print(err)
        self.assertTrue(passed)

    def test_Problem2D_CC(self, tolerance=0.05):
        problem = DC.Problem2D_CC(self.mesh, sigma=self.sigma)
        problem.Solver = self.Solver
        problem.pair(self.survey)
        data = self.survey.dpred()
        err = (
            np.linalg.norm((data-self.data_ana)/self.data_ana)**2 /
            self.data_ana.size
        )
        if err < tolerance:
            passed = True
            print(">> DC analytic test for DPP Problem2D_CC is passed")
        else:
            passed = False
            print(">> DC analytic test for DPP Problem2D_CC is failed")
            print(err)
            if self.plotIt:
                plt.plot(self.data_ana)
                plt.plot(data, 'k.')
                plt.show()
        self.assertTrue(passed)


class DCProblemAnalyticTests_PP(unittest.TestCase):

    def setUp(self):
        # Note: Pole-Pole requires bigger boundary to obtain good accuracy.
        # One can use greater padding rate. Here 1.5 is used.
        cs = 12.5
        hx = [(cs, 7, -1.5), (cs, 61), (cs, 7, 1.5)]
        hy = [(cs, 7, -1.5), (cs, 20)]
        mesh = Mesh.TensorMesh([hx, hy], x0="CN")
        sighalf = 1e-2
        sigma = np.ones(mesh.nC)*sighalf
        x = np.linspace(0, 250., 20)
        M = Utils.ndgrid(x-12.5, np.r_[0.])
        A0loc = np.r_[-150, 0.]
        rxloc = np.c_[M, np.zeros(20)]
        data_ana = EM.Analytics.DCAnalytic_Pole_Pole(
                    np.r_[A0loc, 0.],
                    rxloc, sighalf, earth_type="halfspace")

        rx = DC.Rx.Pole_ky(M)
        src0 = DC.Src.Pole([rx], A0loc)
        survey = DC.Survey_ky([src0])

        self.survey = survey
        self.mesh = mesh
        self.sigma = sigma
        self.data_ana = data_ana

        try:
            from pymatsolver import PardisoSolver
            self.Solver = PardisoSolver
        except ImportError:
            self.Solver = SolverLU

    def test_Problem2D_CC(self, tolerance=0.05):
        problem = DC.Problem2D_CC(self.mesh, sigma=self.sigma, bc_type="Mixed")
        problem.Solver = self.Solver
        problem.pair(self.survey)
        data = self.survey.dpred()
        err = (
            np.linalg.norm((data-self.data_ana)/self.data_ana)**2 /
            self.data_ana.size
        )
        if err < tolerance:
            passed = True
            print(">> DC analytic test for PP Problem2D_CC is passed")
        else:
            passed = False
            print(">> DC analytic test for PP Problem2D_CC is failed")
            print(err)
        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
