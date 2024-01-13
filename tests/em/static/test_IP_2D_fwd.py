import unittest
import discretize

from SimPEG import utils, maps
import numpy as np
from SimPEG.electromagnetics import resistivity as dc
from SimPEG.electromagnetics import induced_polarization as ip

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


class IPProblemAnalyticTests(unittest.TestCase):
    def setUp(self):
        cs = 12.5
        hx = [(cs, 7, -1.3), (cs, 61), (cs, 7, 1.3)]
        hy = [(cs, 7, -1.3), (cs, 20)]
        mesh = discretize.TensorMesh([hx, hy], x0="CN")

        x = np.linspace(-200, 200.0, 20)
        M = utils.ndgrid(x - 12.5, np.r_[0.0])
        N = utils.ndgrid(x + 12.5, np.r_[0.0])
        A0loc = np.r_[-150, 0.0]
        A1loc = np.r_[-130, 0.0]
        B0loc = np.r_[-130, 0.0]
        B1loc = np.r_[-110, 0.0]

        rx = dc.Rx.Dipole(M, N)
        src0 = dc.Src.Dipole([rx], A0loc, B0loc)
        src1 = dc.Src.Dipole([rx], A1loc, B1loc)

        src0_ip = dc.Src.Dipole([rx], A0loc, B0loc)
        src1_ip = dc.Src.Dipole([rx], A1loc, B1loc)

        source_lists = [src0, src1]
        source_lists_ip = [src0_ip, src1_ip]
        surveyDC = dc.Survey([src0, src1])

        sigmaInf = np.ones(mesh.nC) * 1.0
        blkind = utils.model_builder.get_indices_sphere(np.r_[0, -150], 40, mesh.gridCC)

        eta = np.zeros(mesh.nC)
        eta[blkind] = 0.1
        sigma0 = sigmaInf * (1.0 - eta)

        self.surveyDC = surveyDC
        self.mesh = mesh
        self.sigmaInf = sigmaInf
        self.sigma0 = sigma0
        self.source_lists = source_lists
        self.source_lists_ip = source_lists_ip
        self.eta = eta

    def test_Simulation2DNodal(self):
        problemDC = dc.Simulation2DNodal(
            self.mesh, survey=self.surveyDC, sigmaMap=maps.IdentityMap(self.mesh)
        )
        problemDC.solver = Solver
        data0 = problemDC.dpred(self.sigma0)
        datainf = problemDC.dpred(self.sigmaInf)

        surveyIP = ip.Survey(self.source_lists_ip)

        problemIP = ip.Simulation2DNodal(
            self.mesh,
            survey=surveyIP,
            sigma=self.sigmaInf,
            etaMap=maps.IdentityMap(self.mesh),
        )
        problemIP.solver = Solver

        data_full = data0 - datainf
        data = problemIP.dpred(self.eta)
        err = np.linalg.norm((data - data_full) / data_full) ** 2 / data_full.size
        if err < 0.05:
            passed = True
            print(">> IP forward test for Simulation2DNodal is passed")
            print(err)
        else:
            passed = False
            print(">> IP forward test for Simulation2DNodal is failed")
        self.assertTrue(passed)

    def test_Simulation2DCellCentered(self):
        problemDC = dc.Simulation2DCellCentered(
            self.mesh, survey=self.surveyDC, rhoMap=maps.IdentityMap(self.mesh)
        )
        problemDC.solver = Solver
        data0 = problemDC.dpred(1.0 / self.sigma0)
        finf = problemDC.fields(1.0 / self.sigmaInf)
        datainf = problemDC.dpred(1.0 / self.sigmaInf, f=finf)

        surveyIP = ip.Survey(self.source_lists_ip)

        problemIP = ip.Simulation2DCellCentered(
            self.mesh,
            survey=surveyIP,
            rho=1.0 / self.sigmaInf,
            etaMap=maps.IdentityMap(self.mesh),
        )
        problemIP.solver = Solver
        data_full = data0 - datainf
        data = problemIP.dpred(self.eta)
        err = np.linalg.norm((data - data_full) / data_full) ** 2 / data_full.size
        if err < 0.05:
            passed = True
            print(">> IP forward test for Simulation2DCellCentered is passed")
        else:
            import matplotlib.pyplot as plt

            passed = False
            print(">> IP forward test for Simulation2DCellCentered is failed")
            print(err)
            plt.plot(data_full)
            plt.plot(data, "k.")
            plt.show()

        self.assertTrue(passed)


class ApparentChargeability2DTest(unittest.TestCase):
    def setUp(self):
        cs = 12.5
        hx = [(cs, 7, -1.3), (cs, 61), (cs, 7, 1.3)]
        hy = [(cs, 7, -1.3), (cs, 20)]
        mesh = discretize.TensorMesh([hx, hy], x0="CN")

        x = np.linspace(-200, 200.0, 20)
        M = utils.ndgrid(x - 12.5, np.r_[0.0])
        N = utils.ndgrid(x + 12.5, np.r_[0.0])
        A0loc = np.r_[-150, 0.0]
        A1loc = np.r_[-130, 0.0]
        B0loc = np.r_[-130, 0.0]
        B1loc = np.r_[-110, 0.0]

        rx = dc.Rx.Dipole(M, N)
        src0 = dc.Src.Dipole([rx], A0loc, B0loc)
        src1 = dc.Src.Dipole([rx], A1loc, B1loc)

        survey_dc = dc.Survey([src0, src1])

        rx_ip = dc.Rx.Dipole(M, N, data_type="apparent_chargeability")
        src0_ip = dc.Src.Dipole([rx_ip], A0loc, B0loc)
        src1_ip = dc.Src.Dipole([rx_ip], A1loc, B1loc)

        survey_ip = ip.Survey([src0_ip, src1_ip])

        sigmaInf = np.ones(mesh.nC) * 1.0
        blkind = utils.model_builder.get_indices_sphere(np.r_[0, -150], 40, mesh.gridCC)

        eta = np.zeros(mesh.nC)
        eta[blkind] = 0.05
        sigma0 = sigmaInf * (1.0 - eta)

        self.survey_dc = survey_dc
        self.survey_ip = survey_ip
        self.mesh = mesh
        self.sigmaInf = sigmaInf
        self.sigma0 = sigma0
        self.eta = eta

    def test_Simulation2DNodal(self):
        simDC = dc.Simulation2DNodal(
            self.mesh,
            sigmaMap=maps.IdentityMap(self.mesh),
            solver=Solver,
            survey=self.survey_dc,
        )
        data0 = simDC.dpred(self.sigma0)
        datainf = simDC.dpred(self.sigmaInf)
        data_full = (data0 - datainf) / datainf

        simIP = ip.Simulation2DNodal(
            self.mesh,
            sigma=self.sigmaInf,
            etaMap=maps.IdentityMap(self.mesh),
            solver=Solver,
            survey=self.survey_ip,
        )
        data = simIP.dpred(self.eta)

        simIP_store = ip.Simulation2DNodal(
            self.mesh,
            sigma=self.sigmaInf,
            etaMap=maps.IdentityMap(self.mesh),
            solver=Solver,
            survey=self.survey_ip,
            storeJ=True,
        )
        data2 = simIP_store.dpred(self.eta)

        np.testing.assert_allclose(data, data2)

        np.testing.assert_allclose(simIP._scale, 1.0 / datainf)

        err = np.linalg.norm((data - data_full) / data_full) ** 2 / data_full.size
        if err > 0.05:
            import matplotlib.pyplot as plt

            plt.plot(data_full)
            plt.plot(data, "k.")
            plt.show()

        self.assertLess(err, 0.05)

    def test_Simulation2DCellCentered(self):
        simDC = dc.Simulation2DCellCentered(
            self.mesh,
            sigmaMap=maps.IdentityMap(self.mesh),
            solver=Solver,
            survey=self.survey_dc,
        )
        data0 = simDC.dpred(self.sigma0)
        datainf = simDC.dpred(self.sigmaInf)
        data_full = (data0 - datainf) / datainf

        simIP = ip.Simulation2DCellCentered(
            self.mesh,
            sigma=self.sigmaInf,
            etaMap=maps.IdentityMap(self.mesh),
            solver=Solver,
            survey=self.survey_ip,
        )
        data = simIP.dpred(self.eta)

        np.testing.assert_allclose(simIP._scale, 1.0 / datainf)

        err = np.linalg.norm((data - data_full) / data_full) ** 2 / data_full.size
        if err > 0.05:
            import matplotlib.pyplot as plt

            plt.plot(data_full)
            plt.plot(data, "k.")
            plt.show()

        self.assertLess(err, 0.05)


if __name__ == "__main__":
    unittest.main()
