import unittest

import numpy as np
import discretize

from SimPEG import utils, maps
from SimPEG.electromagnetics import resistivity as dc
from SimPEG.electromagnetics import induced_polarization as ip

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


class IPProblemAnalyticTests(unittest.TestCase):
    def setUp(self):
        cs = 12.5
        npad = 2
        hx = [(cs, npad, -1.3), (cs, 21), (cs, npad, 1.3)]
        hy = [(cs, npad, -1.3), (cs, 21), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, 20)]
        mesh = discretize.TensorMesh([hx, hy, hz], x0="CCN")

        x = mesh.cell_centers_x[
            (mesh.cell_centers_x > -80.0) & (mesh.cell_centers_x < 80.0)
        ]
        y = mesh.cell_centers_y[
            (mesh.cell_centers_y > -80.0) & (mesh.cell_centers_y < 80.0)
        ]
        Aloc = np.r_[-100.0, 0.0, 0.0]
        Bloc = np.r_[100.0, 0.0, 0.0]
        M = utils.ndgrid(x - 12.5, y, np.r_[0.0])
        N = utils.ndgrid(x + 12.5, y, np.r_[0.0])
        radius = 50.0
        xc = np.r_[0.0, 0.0, -100]
        blkind = utils.model_builder.get_indices_sphere(xc, radius, mesh.gridCC)
        sigmaInf = np.ones(mesh.nC) * 1e-2
        eta = np.zeros(mesh.nC)
        eta[blkind] = 0.1
        sigma0 = sigmaInf * (1.0 - eta)

        rx = dc.receivers.Dipole(M, N)
        src = dc.sources.Dipole([rx], Aloc, Bloc)
        surveyDC = dc.survey.Survey([src])

        self.surveyDC = surveyDC
        self.mesh = mesh
        self.sigmaInf = sigmaInf
        self.sigma0 = sigma0
        self.src = src
        self.eta = eta

    def test_Simulation3DNodal(self):
        simulationdc = dc.simulation.Simulation3DNodal(
            mesh=self.mesh, survey=self.surveyDC, sigmaMap=maps.IdentityMap(self.mesh)
        )
        simulationdc.solver = Solver
        data0 = simulationdc.dpred(self.sigma0)
        finf = simulationdc.fields(self.sigmaInf)
        datainf = simulationdc.dpred(self.sigmaInf, f=finf)
        surveyip = ip.survey.Survey([self.src])
        simulationip = ip.simulation.Simulation3DNodal(
            mesh=self.mesh,
            survey=surveyip,
            sigma=self.sigmaInf,
            etaMap=maps.IdentityMap(self.mesh),
            Ainv=simulationdc.Ainv,
            _f=finf,
        )
        simulationip.solver = Solver
        data_full = data0 - datainf
        data = simulationip.dpred(self.eta)
        err = np.linalg.norm((data - data_full) / data_full) ** 2 / data_full.size
        if err < 0.05:
            passed = True
            print(">> IP forward test for Simulation3DNodal is passed")
        else:
            passed = False
            print(">> IP forward test for Simulation3DNodal is failed")
        self.assertTrue(passed)

    def test_Simulation3DCellCentered(self):
        simulationdc = dc.simulation.Simulation3DCellCentered(
            mesh=self.mesh, survey=self.surveyDC, sigmaMap=maps.IdentityMap(self.mesh)
        )
        simulationdc.solver = Solver
        data0 = simulationdc.dpred(self.sigma0)
        finf = simulationdc.fields(self.sigmaInf)
        datainf = simulationdc.dpred(self.sigmaInf, f=finf)
        surveyip = ip.survey.Survey([self.src])
        simulationip = ip.simulation.Simulation3DCellCentered(
            mesh=self.mesh,
            survey=surveyip,
            rho=1.0 / self.sigmaInf,
            etaMap=maps.IdentityMap(self.mesh),
            Ainv=simulationdc.Ainv,
            _f=finf,
        )
        simulationip.solver = Solver
        data_full = data0 - datainf
        data = simulationip.dpred(self.eta)
        err = np.linalg.norm((data - data_full) / data_full) ** 2 / data_full.size
        if err < 0.05:
            passed = True
            print(">> IP forward test for Simulation3DCellCentered is passed")
        else:
            passed = False
            print(">> IP forward test for Simulation3DCellCentered is failed")
        self.assertTrue(passed)


class ApparentChargeability3DTest(unittest.TestCase):
    def setUp(self):
        cs = 12.5
        npad = 2
        hx = [(cs, npad, -1.3), (cs, 21), (cs, npad, 1.3)]
        hy = [(cs, npad, -1.3), (cs, 21), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, 20)]
        mesh = discretize.TensorMesh([hx, hy, hz], x0="CCN")

        x = mesh.cell_centers_x[
            (mesh.cell_centers_x > -80.0) & (mesh.cell_centers_x < 80.0)
        ]
        y = mesh.cell_centers_y[
            (mesh.cell_centers_y > -80.0) & (mesh.cell_centers_y < 80.0)
        ]
        Aloc = np.r_[-100.0, 0.0, 0.0]
        Bloc = np.r_[100.0, 0.0, 0.0]
        M = utils.ndgrid(x - 12.5, y, np.r_[0.0])
        N = utils.ndgrid(x + 12.5, y, np.r_[0.0])
        radius = 50.0
        xc = np.r_[0.0, 0.0, -100]
        blkind = utils.model_builder.get_indices_sphere(xc, radius, mesh.gridCC)
        sigmaInf = np.ones(mesh.nC) * 1e-2
        eta = np.zeros(mesh.nC)
        eta[blkind] = 0.1
        sigma0 = sigmaInf * (1.0 - eta)

        rx = dc.receivers.Dipole(M, N)
        src = dc.sources.Dipole([rx], Aloc, Bloc)
        survey_dc = dc.survey.Survey([src])

        rx_ip = dc.receivers.Dipole(M, N, data_type="apparent_chargeability")
        src_ip = dc.sources.Dipole([rx_ip], Aloc, Bloc)
        survey_ip = ip.Survey([src_ip])

        self.survey_dc = survey_dc
        self.survey_ip = survey_ip
        self.mesh = mesh
        self.sigmaInf = sigmaInf
        self.sigma0 = sigma0
        self.src = src
        self.eta = eta

    def test_Simulation3DNodal(self):
        simulationdc = dc.simulation.Simulation3DNodal(
            self.mesh,
            sigmaMap=maps.IdentityMap(self.mesh),
            solver=Solver,
            survey=self.survey_dc,
        )
        data0 = simulationdc.dpred(self.sigma0)
        datainf = simulationdc.dpred(self.sigmaInf)
        data_full = (data0 - datainf) / datainf

        simulationip = ip.simulation.Simulation3DNodal(
            mesh=self.mesh,
            survey=self.survey_ip,
            sigma=self.sigmaInf,
            etaMap=maps.IdentityMap(self.mesh),
            Ainv=simulationdc.Ainv,
            solver=Solver,
        )
        data = simulationip.dpred(self.eta)

        simulationip_stored = ip.simulation.Simulation3DNodal(
            mesh=self.mesh,
            survey=self.survey_ip,
            sigma=self.sigmaInf,
            etaMap=maps.IdentityMap(self.mesh),
            Ainv=simulationdc.Ainv,
            solver=Solver,
            storeJ=True,
        )
        data2 = simulationip_stored.dpred(self.eta)

        np.testing.assert_allclose(data, data2)

        np.testing.assert_allclose(simulationip._scale, 1.0 / datainf)

        err = np.linalg.norm((data - data_full) / data_full) ** 2 / data_full.size
        if err > 0.05:
            import matplotlib.pyplot as plt

            plt.plot(data_full)
            plt.plot(data, "k.")
            plt.show()

        self.assertLess(err, 0.05)

    def test_Simulation3DCellCentered(self):
        simulationdc = dc.simulation.Simulation3DCellCentered(
            self.mesh,
            sigmaMap=maps.IdentityMap(self.mesh),
            solver=Solver,
            survey=self.survey_dc,
        )
        data0 = simulationdc.dpred(self.sigma0)
        datainf = simulationdc.dpred(self.sigmaInf)
        data_full = (data0 - datainf) / datainf

        simulationip = ip.simulation.Simulation3DCellCentered(
            mesh=self.mesh,
            survey=self.survey_ip,
            sigma=self.sigmaInf,
            etaMap=maps.IdentityMap(self.mesh),
            Ainv=simulationdc.Ainv,
            solver=Solver,
        )
        data = simulationip.dpred(self.eta)

        np.testing.assert_allclose(simulationip._scale, 1.0 / datainf)

        err = np.linalg.norm((data - data_full) / data_full) ** 2 / data_full.size
        if err > 0.05:
            import matplotlib.pyplot as plt

            plt.plot(data_full)
            plt.plot(data, "k.")
            plt.show()

        self.assertLess(err, 0.05)


if __name__ == "__main__":
    unittest.main()
