import numpy as np
import unittest
import matplotlib.pyplot as plt

from discretize import TensorMesh

from SimPEG import utils, SolverLU
from SimPEG.electromagnetics import resistivity as dc
from SimPEG.electromagnetics import analytics


class DCProblemAnalyticTests_PDP(unittest.TestCase):
    def setUp(self):

        cs = 12.5
        hx = [(cs, 7, -1.3), (cs, 61), (cs, 7, 1.3)]
        hy = [(cs, 7, -1.3), (cs, 20)]
        mesh = TensorMesh([hx, hy], x0="CN")
        sighalf = 1e-2
        sigma = np.ones(mesh.nC) * sighalf
        x = np.linspace(-135, 250.0, 20)
        M = utils.ndgrid(x - 12.5, np.r_[0.0])
        N = utils.ndgrid(x + 12.5, np.r_[0.0])
        A0loc = np.r_[-150, 0.0]
        # A1loc = np.r_[-130, 0.]
        rxloc = [np.c_[M, np.zeros(20)], np.c_[N, np.zeros(20)]]
        data_ana = analytics.DCAnalytic_Pole_Dipole(
            np.r_[A0loc, 0.0], rxloc, sighalf, earth_type="halfspace"
        )

        rx = dc.receivers.Dipole(M, N)
        src0 = dc.sources.Pole([rx], A0loc)
        survey = dc.Survey_ky([src0])

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

    def test_Simulation2DNodal(self, tolerance=0.05):

        simulation = dc.simulation_2d.Simulation2DNodal(
            self.mesh, survey=self.survey, sigma=self.sigma
        )
        simulation.Solver = self.Solver
        data = simulation.dpred()
        err = (
            np.linalg.norm((data - self.data_ana) / self.data_ana) ** 2
            / self.data_ana.size
        )
        if err < tolerance:
            passed = True
            print(">> DC analytic test for PDP Simulation2DNodal is passed")
        else:
            print(err)
            passed = False
            print(">> DC analytic test for PDP Simulation2DNodal is failed")
        self.assertTrue(passed)

    def test_Simulation2DCellCentered(self, tolerance=0.05):
        simulation = dc.simulation_2d.Simulation2DCellCentered(
            self.mesh, survey=self.survey, sigma=self.sigma
        )
        simulation.Solver = self.Solver
        data = simulation.dpred()
        err = (
            np.linalg.norm((data - self.data_ana) / self.data_ana) ** 2
            / self.data_ana.size
        )
        if err < tolerance:
            passed = True
            print(">> DC analytic test for PDP Simulation2DCellCentered is passed")
        else:
            print(err)
            passed = False
            print(">> DC analytic test for PDP Simulation2DCellCentered is failed")
        self.assertTrue(passed)


class DCProblemAnalyticTests_DPP(unittest.TestCase):
    def setUp(self):

        cs = 12.5
        hx = [(cs, 7, -1.3), (cs, 61), (cs, 7, 1.3)]
        hy = [(cs, 7, -1.3), (cs, 20)]
        mesh = TensorMesh([hx, hy], x0="CN")
        sighalf = 1e-2
        sigma = np.ones(mesh.nC) * sighalf
        x = np.linspace(0, 250.0, 20)
        M = utils.ndgrid(x - 12.5, np.r_[0.0])
        N = utils.ndgrid(x + 12.5, np.r_[0.0])
        A0loc = np.r_[-150, 0.0]
        A1loc = np.r_[-125, 0.0]
        rxloc = np.c_[M, np.zeros(20)]
        data_ana = analytics.DCAnalytic_Dipole_Pole(
            [np.r_[A0loc, 0.0], np.r_[A1loc, 0.0]],
            rxloc,
            sighalf,
            earth_type="halfspace",
        )

        rx = dc.receivers.Pole(M)
        src0 = dc.sources.Dipole([rx], A0loc, A1loc)
        survey = dc.survey.Survey_ky([src0])

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

    def test_Simulation2DNodal(self, tolerance=0.05):

        simulation = dc.simulation_2d.Simulation2DNodal(
            self.mesh, survey=self.survey, sigma=self.sigma
        )
        simulation.Solver = self.Solver
        simulation.pair(self.survey)
        data = simulation.dpred()
        err = (
            np.linalg.norm((data - self.data_ana) / self.data_ana) ** 2
            / self.data_ana.size
        )
        if err < tolerance:
            passed = True
            print(">> DC analytic test for DPP Simulation2DNodal is passed")
            if self.plotIt:
                plt.plot(self.data_ana)
                plt.plot(data, "k.")
                plt.show()
        else:
            passed = False
            print(">> DC analytic test for DPP Simulation2DNodal is failed")
            print(err)
        self.assertTrue(passed)

    def test_Simulation2DCellCentered(self, tolerance=0.05):
        simulation = dc.simulation_2d.Simulation2DCellCentered(
            self.mesh, survey=self.survey, sigma=self.sigma
        )
        simulation.Solver = self.Solver
        data = simulation.dpred()
        err = (
            np.linalg.norm((data - self.data_ana) / self.data_ana) ** 2
            / self.data_ana.size
        )
        if err < tolerance:
            passed = True
            print(">> DC analytic test for DPP Simulation2DCellCentered is passed")
        else:
            passed = False
            print(">> DC analytic test for DPP Simulation2DCellCentered is failed")
            print(err)
            if self.plotIt:
                plt.plot(self.data_ana)
                plt.plot(data, "k.")
                plt.show()
        self.assertTrue(passed)


class DCProblemAnalyticTests_PP(unittest.TestCase):
    def setUp(self):
        # Note: Pole-Pole requires bigger boundary to obtain good accuracy.
        # One can use greater padding rate. Here 1.5 is used.
        cs = 12.5
        hx = [(cs, 7, -1.5), (cs, 61), (cs, 7, 1.5)]
        hy = [(cs, 7, -1.5), (cs, 20)]
        mesh = TensorMesh([hx, hy], x0="CN")
        sighalf = 1e-2
        sigma = np.ones(mesh.nC) * sighalf
        x = np.linspace(0, 250.0, 20)
        M = utils.ndgrid(x - 12.5, np.r_[0.0])
        A0loc = np.r_[-150, 0.0]
        rxloc = np.c_[M, np.zeros(20)]
        data_ana = analytics.DCAnalytic_Pole_Pole(
            np.r_[A0loc, 0.0], rxloc, sighalf, earth_type="halfspace"
        )

        rx = dc.receivers.Pole(M)
        src0 = dc.sources.Pole([rx], A0loc)
        survey = dc.survey.Survey_ky([src0])

        self.survey = survey
        self.mesh = mesh
        self.sigma = sigma
        self.data_ana = data_ana

        try:
            from pymatsolver import PardisoSolver

            self.Solver = PardisoSolver
        except ImportError:
            self.Solver = SolverLU

    def test_Simulation2DCellCentered(self, tolerance=0.05):
        simulation = dc.simulation_2d.Simulation2DCellCentered(
            self.mesh, survey=self.survey, sigma=self.sigma, bc_type="Mixed"
        )
        simulation.Solver = self.Solver
        data = simulation.dpred()
        err = (
            np.linalg.norm((data - self.data_ana) / self.data_ana) ** 2
            / self.data_ana.size
        )
        if err < tolerance:
            passed = True
            print(">> DC analytic test for PP Simulation2DCellCentered is passed")
        else:
            passed = False
            print(">> DC analytic test for PP Simulation2DCellCentered is failed")
            print(err)
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
