import numpy as np
import unittest

from discretize import TensorMesh

from SimPEG import utils, SolverLU
from SimPEG.electromagnetics import resistivity as dc
from SimPEG.electromagnetics import analytics


class DCProblemAnalyticTests_DPDP(unittest.TestCase):
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
        A1loc = np.r_[-130, 0.0]
        rxloc = [np.c_[M, np.zeros(20)], np.c_[N, np.zeros(20)]]
        data_ana_A = analytics.DCAnalytic_Pole_Dipole(
            np.r_[A0loc, 0.0], rxloc, sighalf, earth_type="halfspace"
        )

        data_ana_b = analytics.DCAnalytic_Pole_Dipole(
            np.r_[A1loc, 0.0], rxloc, sighalf, earth_type="halfspace"
        )
        data_ana = data_ana_A - data_ana_b

        rx = dc.receivers.Dipole(M, N)
        src0 = dc.sources.Dipole([rx], A0loc, A1loc)
        survey = dc.Survey([src0])

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
            self.mesh, survey=self.survey, sigma=self.sigma, solver=self.Solver,
        )
        data = simulation.dpred()
        err = (
            np.linalg.norm((data - self.data_ana) / self.data_ana) ** 2
            / self.data_ana.size
        )
        print(f"DPDP N err: {err}")
        self.assertLess(err, tolerance)


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
        survey = dc.Survey([src0])

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
            self.mesh, survey=self.survey, sigma=self.sigma, solver=self.Solver,
        )
        data = simulation.dpred()
        err = (
            np.linalg.norm((data - self.data_ana) / self.data_ana) ** 2
            / self.data_ana.size
        )
        print(f"PDP N err: {err}")
        self.assertLess(err, tolerance)

    def test_Simulation2DCellCentered(self, tolerance=0.05):
        simulation = dc.simulation_2d.Simulation2DCellCentered(
            self.mesh, survey=self.survey, sigma=self.sigma, solver=self.Solver,
        )
        data = simulation.dpred()
        err = (
            np.linalg.norm((data - self.data_ana) / self.data_ana) ** 2
            / self.data_ana.size
        )
        print(f"PDP CC err: {err}")
        self.assertLess(err, tolerance)


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
        survey = dc.survey.Survey([src0])

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
            self.mesh, survey=self.survey, sigma=self.sigma, solver=self.Solver,
        )
        data = simulation.dpred()
        err = (
            np.linalg.norm((data - self.data_ana) / self.data_ana) ** 2
            / self.data_ana.size
        )
        print(f"DPP N err: {err}")
        self.assertLess(err, tolerance)

    def test_Simulation2DCellCentered(self, tolerance=0.05):
        simulation = dc.simulation_2d.Simulation2DCellCentered(
            self.mesh, survey=self.survey, sigma=self.sigma, solver=self.Solver,
        )
        data = simulation.dpred()
        err = (
            np.linalg.norm((data - self.data_ana) / self.data_ana) ** 2
            / self.data_ana.size
        )
        print(f"DPP CC err: {err}")
        self.assertLess(err, tolerance)


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
        survey = dc.survey.Survey([src0])

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
            self.mesh,
            survey=self.survey,
            sigma=self.sigma,
            bc_type="Mixed",
            solver=self.Solver,
        )
        data = simulation.dpred()
        err = (
            np.linalg.norm((data - self.data_ana) / self.data_ana) ** 2
            / self.data_ana.size
        )
        print(f"PP CC err: {err}")
        self.assertLess(err, tolerance)


class DCProblemAnalyticTests_DPField(unittest.TestCase):
    def setUp(self):

        cs = 12.5
        hx = [(cs, 7, -1.3), (cs, 61), (cs, 7, 1.3)]
        hy = [(cs, 7, -1.3), (cs, 20)]
        mesh = TensorMesh([hx, hy], x0="CN")
        sighalf = 1e-2
        sigma = np.ones(mesh.nC) * sighalf
        A0loc = np.r_[-31.25, 0.0]
        A1loc = np.r_[31.25, 0.0]

        rxloc = np.c_[mesh.gridN, np.zeros(mesh.nN)]
        data_ana = analytics.DCAnalytic_Dipole_Pole(
            [np.r_[A0loc, 0.0], np.r_[A1loc, 0.0]],
            rxloc,
            sighalf,
            earth_type="halfspace",
        )

        src0 = dc.sources.Dipole([], A0loc, A1loc)
        survey = dc.survey.Survey([src0])

        # determine comparison locations
        ROI_large_BNW = np.array([-200, -100])
        ROI_large_TSE = np.array([200, 0])
        ROI_largeInds = utils.model_builder.getIndicesBlock(
            ROI_large_BNW, ROI_large_TSE, mesh.gridN
        )[0]
        # print(ROI_largeInds.shape)

        ROI_small_BNW = np.array([-50, -25])
        ROI_small_TSE = np.array([50, 0])
        ROI_smallInds = utils.model_builder.getIndicesBlock(
            ROI_small_BNW, ROI_small_TSE, mesh.gridN
        )[0]
        # print(ROI_smallInds.shape)

        ROI_inds = np.setdiff1d(ROI_largeInds, ROI_smallInds)

        self.data_ana = data_ana
        self.survey = survey
        self.mesh = mesh
        self.sigma = sigma
        self.plotIt = False
        self.ROI_inds = ROI_inds

        try:
            from pymatsolver import PardisoSolver

            self.Solver = PardisoSolver
        except ImportError:
            self.Solver = SolverLU

    def test_Simulation2DCellCentered(self, tolerance=0.05):
        simulation = dc.simulation_2d.Simulation2DCellCentered(
            self.mesh, survey=self.survey, sigma=self.sigma, solver=self.Solver,
        )
        field = simulation.fields(self.sigma)

        # just test if we can get each property of the field
        field[:, "phi"][:, 0]
        field[:, "j"]
        field[:, "e"]
        field[:, "charge"]
        field[:, "charge_density"]
        print("got fields CC")

    def test_Simulation2DNodal(self, tolerance=0.05):

        simulation = dc.simulation_2d.Simulation2DNodal(
            self.mesh, survey=self.survey, sigma=self.sigma, solver=self.Solver,
        )
        field = simulation.fields(self.sigma)
        data = field[:, "phi"][:, 0]

        # also test if we can get the other things charge and charge_density
        field[:, "j"]
        field[:, "e"]
        field[:, "charge"]
        field[:, "charge_density"]
        print("got fields N")

        ROI_inds = self.ROI_inds
        diff_norm = np.linalg.norm((data[ROI_inds] - self.data_ana[ROI_inds]))
        err = diff_norm / np.linalg.norm(self.data_ana[ROI_inds])
        print(f"DP N Fields err: {err}")
        self.assertLess(err, tolerance)


if __name__ == "__main__":
    unittest.main()
