import numpy as np
import unittest

from discretize import TensorMesh

from simpeg import utils, SolverLU
from simpeg.electromagnetics import resistivity as dc
from simpeg.electromagnetics import analytics


class DCProblemAnalyticTests_DPDP(unittest.TestCase):
    def setUp(self):
        npad = 10
        cs = 12.5
        hx = [(cs, npad, -1.4), (cs, 61), (cs, npad, 1.4)]
        hy = [(cs, npad, -1.4), (cs, 20)]
        mesh = TensorMesh([hx, hy], x0="CN")
        sighalf = 1e-2
        sigma = np.ones(mesh.nC) * sighalf
        x = mesh.cell_centers_x[
            np.logical_and(mesh.cell_centers_x > -150, mesh.cell_centers_x < 250)
        ]
        M = utils.ndgrid(x, np.r_[0.0])
        N = utils.ndgrid(x + 12.5 * 4, np.r_[0.0])
        A0loc = np.r_[-200, 0.0]
        A1loc = np.r_[-250, 0.0]
        rxloc = [np.c_[M, np.zeros(x.size)], np.c_[N, np.zeros(x.size)]]
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

            self.solver = Pardiso
        except ImportError:
            self.solver = SolverLU

    def test_Simulation2DNodal(self, tolerance=0.05):
        simulation = dc.simulation_2d.Simulation2DNodal(
            self.mesh,
            survey=self.survey,
            sigma=self.sigma,
            solver=self.solver,
            bc_type="Robin",
        )
        data = simulation.dpred()
        err = np.sqrt(
            np.linalg.norm((data - self.data_ana) / self.data_ana) ** 2
            / self.data_ana.size
        )
        print(f"DPDP N err: {err}")
        self.assertLess(err, tolerance)

    def test_Simulation2DCellCentered(self, tolerance=0.05):
        simulation = dc.simulation_2d.Simulation2DCellCentered(
            self.mesh,
            survey=self.survey,
            sigma=self.sigma,
            solver=self.solver,
            bc_type="Robin",
        )
        data = simulation.dpred()
        err = np.sqrt(
            np.linalg.norm((data - self.data_ana) / self.data_ana) ** 2
            / self.data_ana.size
        )
        print(f"DPDP N err: {err}")
        self.assertLess(err, tolerance)


class DCProblemAnalyticTests_PDP(unittest.TestCase):
    def setUp(self):
        npad = 10
        cs = 12.5
        hx = [(cs, npad, -1.4), (cs, 61), (cs, npad, 1.4)]
        hy = [(cs, npad, -1.4), (cs, 20)]
        mesh = TensorMesh([hx, hy], x0="CN")
        sighalf = 1e-2
        sigma = np.ones(mesh.nC) * sighalf
        x = mesh.cell_centers_x[
            np.logical_and(mesh.cell_centers_x > -150, mesh.cell_centers_x < 250)
        ]
        M = utils.ndgrid(x, np.r_[0.0])
        N = utils.ndgrid(x + 12.5 * 4, np.r_[0.0])
        A0loc = np.r_[-200, 0.0]
        # A1loc = np.r_[-250, 0.0]
        rxloc = [np.c_[M, np.zeros(x.size)], np.c_[N, np.zeros(x.size)]]
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

            self.solver = Pardiso
        except ImportError:
            self.solver = SolverLU

    def test_Simulation2DNodal(self, tolerance=0.05):
        simulation = dc.simulation_2d.Simulation2DNodal(
            self.mesh,
            survey=self.survey,
            sigma=self.sigma,
            solver=self.solver,
            bc_type="Robin",
        )
        data = simulation.dpred()
        err = np.sqrt(
            np.linalg.norm((data - self.data_ana) / self.data_ana) ** 2
            / self.data_ana.size
        )
        print(f"PDP N err: {err}")
        self.assertLess(err, tolerance)

    def test_Simulation2DCellCentered(self, tolerance=0.05):
        simulation = dc.simulation_2d.Simulation2DCellCentered(
            self.mesh,
            survey=self.survey,
            sigma=self.sigma,
            solver=self.solver,
            bc_type="Robin",
        )
        data = simulation.dpred()
        err = np.sqrt(
            np.linalg.norm((data - self.data_ana) / self.data_ana) ** 2
            / self.data_ana.size
        )
        print(f"PDP CC err: {err}")
        self.assertLess(err, tolerance)


class DCProblemAnalyticTests_DPP(unittest.TestCase):
    def setUp(self):
        npad = 10
        cs = 12.5
        hx = [(cs, npad, -1.4), (cs, 61), (cs, npad, 1.4)]
        hy = [(cs, npad, -1.4), (cs, 20)]
        mesh = TensorMesh([hx, hy], x0="CN")
        sighalf = 1e-2
        sigma = np.ones(mesh.nC) * sighalf
        x = mesh.cell_centers_x[
            np.logical_and(mesh.cell_centers_x > -150, mesh.cell_centers_x < 250)
        ]
        M = utils.ndgrid(x, np.r_[0.0])
        A0loc = np.r_[-200, 0.0]
        A1loc = np.r_[-250, 0.0]
        rxloc = np.c_[M, np.zeros(x.size)]
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

            self.solver = PardisoSolver
        except ImportError:
            self.solver = SolverLU

    def test_Simulation2DNodal(self, tolerance=0.05):
        simulation = dc.simulation_2d.Simulation2DNodal(
            self.mesh,
            survey=self.survey,
            sigma=self.sigma,
            solver=self.solver,
            bc_type="Robin",
        )
        data = simulation.dpred()
        err = np.sqrt(
            np.linalg.norm((data - self.data_ana) / self.data_ana) ** 2
            / self.data_ana.size
        )
        print(f"DPP N err: {err}")
        self.assertLess(err, tolerance)

    def test_Simulation2DCellCentered(self, tolerance=0.05):
        simulation = dc.simulation_2d.Simulation2DCellCentered(
            self.mesh,
            survey=self.survey,
            sigma=self.sigma,
            solver=self.solver,
            bc_type="Robin",
        )
        data = simulation.dpred()
        err = np.sqrt(
            np.linalg.norm((data - self.data_ana) / self.data_ana) ** 2
            / self.data_ana.size
        )
        print(f"DPP CC err: {err}")
        self.assertLess(err, tolerance)


class DCProblemAnalyticTests_PP(unittest.TestCase):
    def setUp(self):
        # Note: Pole-Pole requires bigger boundary to obtain good accuracy.
        # One can use greater padding rate. Here 2 is used.
        npad = 10
        cs = 12.5
        hx = [(cs, npad, -2), (cs, 61), (cs, npad, 2)]
        hy = [(cs, npad, -2), (cs, 20)]
        mesh = TensorMesh([hx, hy], x0="CN")
        sighalf = 1e-2
        sigma = np.ones(mesh.nC) * sighalf
        x = mesh.cell_centers_x[
            np.logical_and(mesh.cell_centers_x > -150, mesh.cell_centers_x < 250)
        ]
        M = utils.ndgrid(x, np.r_[0.0])
        # N = utils.ndgrid(x + 12.5*4, np.r_[0.0])
        A0loc = np.r_[-200, 0.0]
        # A1loc = np.r_[-250, 0.0]
        rxloc = np.c_[M, np.zeros(x.size)]
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

            self.solver = PardisoSolver
        except ImportError:
            self.solver = SolverLU

    def test_Simulation2DCellCentered(self, tolerance=0.05):
        simulation = dc.simulation_2d.Simulation2DCellCentered(
            self.mesh,
            survey=self.survey,
            sigma=self.sigma,
            solver=self.solver,
            bc_type="Robin",
        )
        data = simulation.dpred()
        err = np.sqrt(
            np.linalg.norm((data - self.data_ana) / self.data_ana) ** 2
            / self.data_ana.size
        )
        print(f"PP CC err: {err}")
        self.assertLess(err, tolerance)

    def test_Simulation2DNodal(self, tolerance=0.05):
        simulation = dc.simulation_2d.Simulation2DNodal(
            self.mesh,
            survey=self.survey,
            sigma=self.sigma,
            solver=self.solver,
            bc_type="Robin",
        )
        data = simulation.dpred()
        err = np.sqrt(
            np.linalg.norm((data - self.data_ana) / self.data_ana) ** 2
            / self.data_ana.size
        )
        print(f"PP N err: {err}")
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
        ROI_largeInds = utils.model_builder.get_indices_block(
            ROI_large_BNW, ROI_large_TSE, mesh.gridN
        )[0]
        # print(ROI_largeInds.shape)

        ROI_small_BNW = np.array([-50, -25])
        ROI_small_TSE = np.array([50, 0])
        ROI_smallInds = utils.model_builder.get_indices_block(
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

            self.solver = PardisoSolver
        except ImportError:
            self.solver = SolverLU

    def test_Simulation2DCellCentered(self, tolerance=0.05):
        simulation = dc.simulation_2d.Simulation2DCellCentered(
            self.mesh,
            survey=self.survey,
            sigma=self.sigma,
            solver=self.solver,
        )
        field = simulation.fields()

        # just test if we can get each property of the field
        field[:, "phi"][:, 0]
        field[:, "j"]
        field[:, "e"]
        field[:, "charge"]
        field[:, "charge_density"]
        print("got fields CC")

    def test_Simulation2DCellCentered_Dirichlet(self, tolerance=0.05):
        simulation = dc.simulation_2d.Simulation2DCellCentered(
            self.mesh,
            survey=self.survey,
            sigma=self.sigma,
            solver=self.solver,
            bc_type="Dirichlet",
        )
        field = simulation.fields()

        # just test if we can get each property of the field
        field[:, "phi"][:, 0]
        field[:, "j"]
        field[:, "e"]
        field[:, "charge"]
        field[:, "charge_density"]
        print("got fields CC")

    def test_Simulation2DNodal(self, tolerance=0.05):
        simulation = dc.simulation_2d.Simulation2DNodal(
            self.mesh,
            survey=self.survey,
            sigma=self.sigma,
            solver=self.solver,
        )
        field = simulation.fields()
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


class DCSimulationAppResTests(unittest.TestCase):
    def setUp(self):
        npad = 10
        cs = 12.5
        hx = [(cs, npad, -1.4), (cs, 61), (cs, npad, 1.4)]
        hy = [(cs, npad, -1.4), (cs, 20)]
        mesh = TensorMesh([hx, hy], x0="CN")
        sighalf = 1e-2
        sigma = np.ones(mesh.nC) * sighalf
        x = mesh.cell_centers_x[
            np.logical_and(mesh.cell_centers_x > -150, mesh.cell_centers_x < 250)
        ]
        M = utils.ndgrid(x, np.r_[0.0])
        N = utils.ndgrid(x + 12.5 * 4, np.r_[0.0])
        A0loc = np.r_[-200, 0.0]
        A1loc = np.r_[-250, 0.0]
        rx = dc.receivers.Dipole(M, N, data_type="apparent_resistivity")
        src0 = dc.sources.Dipole([rx], A0loc, A1loc)
        survey = dc.Survey([src0])

        self.survey = survey
        self.mesh = mesh
        self.sigma = sigma
        self.sigma_half = sighalf
        self.plotIt = False

        try:
            from pymatsolver import Pardiso

            self.solver = Pardiso
        except ImportError:
            self.solver = SolverLU

    def test_Simulation2DNodal(self, tolerance=0.05):
        simulation = dc.simulation_2d.Simulation2DNodal(
            self.mesh,
            survey=self.survey,
            sigma=self.sigma,
            solver=self.solver,
            bc_type="Robin",
        )
        with self.assertRaises(KeyError):
            data = simulation.dpred()

        self.survey.set_geometric_factor()
        data = simulation.dpred()

        rhohalf = 1.0 / self.sigma_half
        err = np.sqrt(np.linalg.norm((data - rhohalf) / rhohalf) ** 2 / data.size)
        print(f"DPDP N err: {err}")
        self.assertLess(err, tolerance)

    def test_Simulation2DCellCentered(self, tolerance=0.05):
        simulation = dc.simulation_2d.Simulation2DCellCentered(
            self.mesh,
            survey=self.survey,
            sigma=self.sigma,
            solver=self.solver,
            bc_type="Robin",
        )
        with self.assertRaises(KeyError):
            data = simulation.dpred()

        self.survey.set_geometric_factor()
        data = simulation.dpred()

        rhohalf = 1.0 / self.sigma_half
        err = np.sqrt(np.linalg.norm((data - rhohalf) / rhohalf) ** 2 / data.size)
        print(f"DPDP N err: {err}")
        self.assertLess(err, tolerance)


if __name__ == "__main__":
    unittest.main()
