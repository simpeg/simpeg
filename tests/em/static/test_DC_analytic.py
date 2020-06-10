from __future__ import print_function
import unittest
import discretize

from SimPEG import utils
import numpy as np
from SimPEG.electromagnetics import resistivity as dc
from SimPEG.electromagnetics import analytics

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


class DCProblemAnalyticTests(unittest.TestCase):
    def setUp(self):

        cs = 25.0
        npad = 7
        hx = [(cs, npad, -1.3), (cs, 21), (cs, npad, 1.3)]
        hy = [(cs, npad, -1.3), (cs, 21), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, 20)]
        mesh = discretize.TensorMesh([hx, hy, hz], x0="CCN")
        sigma = np.ones(mesh.nC) * 1e-2

        x = mesh.vectorCCx[(mesh.vectorCCx > -155.0) & (mesh.vectorCCx < 155.0)]
        y = mesh.vectorCCy[(mesh.vectorCCy > -155.0) & (mesh.vectorCCy < 155.0)]

        Aloc = np.r_[-200.0, 0.0, 0.0]
        Bloc = np.r_[200.0, 0.0, 0.0]
        M = utils.ndgrid(x - 25.0, y, np.r_[0.0])
        N = utils.ndgrid(x + 25.0, y, np.r_[0.0])
        phiA = analytics.DCAnalytic_Pole_Dipole(
            Aloc, [M, N], 1e-2, earth_type="halfspace"
        )
        phiB = analytics.DCAnalytic_Pole_Dipole(
            Bloc, [M, N], 1e-2, earth_type="halfspace"
        )
        data_ana = phiA - phiB

        rx = dc.receivers.Dipole(M, N)
        src = dc.sources.Dipole([rx], Aloc, Bloc)
        survey = dc.survey.Survey([src])

        self.survey = survey
        self.mesh = mesh
        self.sigma = sigma
        self.data_ana = data_ana

    def test_Simulation3DNodal(self, tolerance=0.2):
        simulation = dc.simulation.Simulation3DNodal(
            self.mesh, survey=self.survey, sigma=self.sigma, solver=Solver
        )
        data = simulation.dpred()
        err = np.linalg.norm(data - self.data_ana) / np.linalg.norm(self.data_ana)
        if err < 0.2:
            print(err)
            passed = True
            print(">> DC analytic test for Simulation3DNodal is passed")
        else:
            print(err)
            passed = False
            print(">> DC analytic test for Simulation3DNodal is failed")
        self.assertTrue(passed)

    def test_Simulation3DCellCentered_Mixed(self, tolerance=0.2):
        simulation = dc.Simulation3DCellCentered(
            self.mesh,
            survey=self.survey,
            sigma=self.sigma,
            bc_type="Mixed",
            solver=Solver,
        )
        data = simulation.dpred()

        err = np.linalg.norm(data - self.data_ana) / np.linalg.norm(self.data_ana)
        if err < tolerance:
            print(err)
            passed = True
            print(">> DC analytic test for Simulation3DCellCentered is passed")
        else:
            print(err)
            passed = False
            print(">> DC analytic test for Simulation3DCellCentered is failed")
        self.assertTrue(passed)

    def test_Simulation3DCellCentered_Neumann(self, tolerance=0.2):
        simulation = dc.simulation.Simulation3DCellCentered(
            self.mesh,
            survey=self.survey,
            sigma=self.sigma,
            bc_type="Neumann",
            solver=Solver,
        )
        data = simulation.dpred()
        err = np.linalg.norm(data - self.data_ana) / np.linalg.norm(self.data_ana)
        if err < tolerance:
            print(err)
            passed = True
            print(">> DC analytic test for Simulation3DCellCentered is passed")
        else:
            print(err)
            passed = False
            print(">> DC analytic test for Simulation3DCellCentered is failed")
        self.assertTrue(passed)


# This is for testing Dirichlet B.C.
# for wholepsace Earth.
class DCProblemAnalyticTests_Dirichlet(unittest.TestCase):
    def setUp(self):

        cs = 25.0
        hx = [(cs, 7, -1.3), (cs, 21), (cs, 7, 1.3)]
        hy = [(cs, 7, -1.3), (cs, 21), (cs, 7, 1.3)]
        hz = [(cs, 7, -1.3), (cs, 20), (cs, 7, -1.3)]
        mesh = discretize.TensorMesh([hx, hy, hz], x0="CCC")
        sigma = np.ones(mesh.nC) * 1e-2

        x = mesh.vectorCCx[(mesh.vectorCCx > -155.0) & (mesh.vectorCCx < 155.0)]
        y = mesh.vectorCCy[(mesh.vectorCCy > -155.0) & (mesh.vectorCCy < 155.0)]

        Aloc = np.r_[-200.0, 0.0, 0.0]
        Bloc = np.r_[200.0, 0.0, 0.0]
        M = utils.ndgrid(x - 25.0, y, np.r_[0.0])
        N = utils.ndgrid(x + 25.0, y, np.r_[0.0])
        phiA = analytics.DCAnalytic_Pole_Dipole(
            Aloc, [M, N], 1e-2, earth_type="wholespace"
        )
        phiB = analytics.DCAnalytic_Pole_Dipole(
            Bloc, [M, N], 1e-2, earth_type="wholespace"
        )
        data_ana = phiA - phiB

        rx = dc.receivers.Dipole(M, N)
        src = dc.sources.Dipole([rx], Aloc, Bloc)
        survey = dc.survey.Survey([src])

        self.survey = survey
        self.mesh = mesh
        self.sigma = sigma
        self.data_ana = data_ana

    def test_Simulation3DCellCentered_Dirichlet(self, tolerance=0.2):
        simulation = dc.simulation.Simulation3DCellCentered(
            self.mesh,
            survey=self.survey,
            sigma=self.sigma,
            bc_type="Dirichlet",
            solver=Solver,
        )

        data = simulation.dpred()
        err = np.linalg.norm(data - self.data_ana) / np.linalg.norm(self.data_ana)
        if err < tolerance:
            print(err)
            passed = True
            print(">> DC analytic test for Simulation3DCellCentered_Dirchlet is passed")
        else:
            print(err)
            passed = False
            print(">> DC analytic test for Simulation3DCellCentered_Dirchlet is failed")
        self.assertTrue(passed)


# This is for Pole-Pole case
class DCProblemAnalyticTests_Mixed(unittest.TestCase):
    def setUp(self):

        cs = 25.0
        hx = [(cs, 7, -1.5), (cs, 21), (cs, 7, 1.5)]
        hy = [(cs, 7, -1.5), (cs, 21), (cs, 7, 1.5)]
        hz = [(cs, 7, -1.5), (cs, 20)]
        mesh = discretize.TensorMesh([hx, hy, hz], x0="CCN")
        sigma = np.ones(mesh.nC) * 1e-2

        x = mesh.vectorCCx[(mesh.vectorCCx > -155.0) & (mesh.vectorCCx < 155.0)]
        y = mesh.vectorCCy[(mesh.vectorCCy > -155.0) & (mesh.vectorCCy < 155.0)]

        Aloc = np.r_[-200.0, 0.0, 0.0]

        M = utils.ndgrid(x, y, np.r_[0.0])
        phiA = analytics.DCAnalytic_Pole_Pole(Aloc, M, 1e-2, earth_type="halfspace")
        data_ana = phiA

        rx = dc.receivers.Pole(M)
        src = dc.sources.Pole([rx], Aloc)
        survey = dc.survey.Survey([src])

        self.survey = survey
        self.mesh = mesh
        self.sigma = sigma
        self.data_ana = data_ana

    def test_Simulation3DCellCentered_Mixed(self, tolerance=0.2):
        simulation = dc.simulation.Simulation3DCellCentered(
            self.mesh,
            survey=self.survey,
            sigma=self.sigma,
            bc_type="Mixed",
            solver=Solver,
        )
        data = simulation.dpred()
        err = np.linalg.norm(data - self.data_ana) / np.linalg.norm(self.data_ana)
        if err < tolerance:
            print(err)
            passed = True
            print(">> DC analytic test for Simulation3DCellCentered_Mixed is passed")
        else:
            print(err)
            passed = False
            print(">> DC analytic test for Simulation3DCellCentered_Mixed is failed")
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
