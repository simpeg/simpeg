from __future__ import print_function
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

        srcLists = [src0, src1]
        srcLists_ip = [src0_ip, src1_ip]
        surveyDC = dc.Survey_ky([src0, src1])

        sigmaInf = np.ones(mesh.nC) * 1.0
        blkind = utils.model_builder.getIndicesSphere(np.r_[0, -150], 40, mesh.gridCC)

        eta = np.zeros(mesh.nC)
        eta[blkind] = 0.1
        sigma0 = sigmaInf * (1.0 - eta)

        self.surveyDC = surveyDC
        self.mesh = mesh
        self.sigmaInf = sigmaInf
        self.sigma0 = sigma0
        self.source_lists = srcLists
        self.source_lists_ip = srcLists_ip
        self.eta = eta

    def test_Simulation2DNodal(self):

        problemDC = dc.Simulation2DNodal(
            self.mesh, sigmaMap=maps.IdentityMap(self.mesh)
        )
        problemDC.Solver = Solver
        problemDC.pair(self.surveyDC)
        data0 = problemDC.dpred(self.sigma0)
        datainf = problemDC.dpred(self.sigmaInf)
        problemIP = ip.Simulation2DNodal(
            self.mesh, sigma=self.sigmaInf, etaMap=maps.IdentityMap(self.mesh),
        )
        problemIP.Solver = Solver
        surveyIP = ip.Survey(self.source_lists_ip)
        problemIP.pair(surveyIP)
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
            self.mesh, rhoMap=maps.IdentityMap(self.mesh)
        )
        problemDC.Solver = Solver
        problemDC.pair(self.surveyDC)
        data0 = problemDC.dpred(1.0 / self.sigma0)
        finf = problemDC.fields(1.0 / self.sigmaInf)
        datainf = problemDC.dpred(1.0 / self.sigmaInf, f=finf)
        problemIP = ip.Simulation2DCellCentered(
            self.mesh, rho=1.0 / self.sigmaInf, etaMap=maps.IdentityMap(self.mesh)
        )
        problemIP.Solver = Solver
        print("\n\n\n")
        print(self.source_lists_ip)
        surveyIP = ip.Survey(self.source_lists_ip)
        problemIP.pair(surveyIP)
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


if __name__ == "__main__":
    unittest.main()
