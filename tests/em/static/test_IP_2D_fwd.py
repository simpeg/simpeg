from __future__ import print_function
import unittest
from SimPEG import Mesh, Utils, Maps
import numpy as np
import SimPEG.EM.Static.DC as DC
import SimPEG.EM.Static.IP as IP
try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


class IPProblemAnalyticTests(unittest.TestCase):

    def setUp(self):

        cs = 12.5
        hx = [(cs, 7, -1.3), (cs, 61), (cs, 7, 1.3)]
        hy = [(cs, 7, -1.3), (cs, 20)]
        mesh = Mesh.TensorMesh([hx, hy], x0="CN")

        x = np.linspace(-200, 200., 20)
        M = Utils.ndgrid(x-12.5, np.r_[0.])
        N = Utils.ndgrid(x+12.5, np.r_[0.])
        A0loc = np.r_[-150, 0.]
        A1loc = np.r_[-130, 0.]
        B0loc = np.r_[-130, 0.]
        B1loc = np.r_[-110, 0.]

        rx = DC.Rx.Dipole_ky(M, N)
        src0 = DC.Src.Dipole([rx], A0loc, B0loc)
        src1 = DC.Src.Dipole([rx], A1loc, B1loc)

        src0_ip = DC.Src.Dipole([rx], A0loc, B0loc)
        src1_ip = DC.Src.Dipole([rx], A1loc, B1loc)

        srcLists = [src0, src1]
        srcLists_ip = [src0_ip, src1_ip]
        surveyDC = DC.Survey_ky([src0, src1])

        sigmaInf = np.ones(mesh.nC) * 1.
        blkind = Utils.ModelBuilder.getIndicesSphere(
            np.r_[0, -150], 40, mesh.gridCC)

        eta = np.zeros(mesh.nC)
        eta[blkind] = 0.1
        sigma0 = sigmaInf * (1.-eta)

        self.surveyDC = surveyDC
        self.mesh = mesh
        self.sigmaInf = sigmaInf
        self.sigma0 = sigma0
        self.srcLists = srcLists
        self.srcLists_ip = srcLists_ip
        self.eta = eta

    def test_Problem2D_N(self):

        problemDC = DC.Problem2D_N(
            self.mesh, sigmaMap=Maps.IdentityMap(self.mesh)
        )
        problemDC.Solver = Solver
        problemDC.pair(self.surveyDC)
        data0 = self.surveyDC.dpred(self.sigma0)
        datainf = self.surveyDC.dpred(self.sigmaInf)
        problemIP = IP.Problem2D_N(
            self.mesh,
            sigma=self.sigmaInf,
            etaMap=Maps.IdentityMap(self.mesh),
        )
        problemIP.Solver = Solver
        surveyIP = IP.Survey(self.srcLists_ip)
        problemIP.pair(surveyIP)
        data_full = data0 - datainf
        data = surveyIP.dpred(self.eta)
        err = np.linalg.norm((data-data_full)/data_full)**2 / data_full.size
        if err < 0.05:
            passed = True
            print(">> IP forward test for Problem2D_N is passed")
            print(err)
        else:
            passed = False
            print(">> IP forward test for Problem2D_N is failed")
        self.assertTrue(passed)

    def test_Problem2D_CC(self):

        problemDC = DC.Problem2D_CC(
            self.mesh, rhoMap=Maps.IdentityMap(self.mesh)
        )
        problemDC.Solver = Solver
        problemDC.pair(self.surveyDC)
        data0 = self.surveyDC.dpred(1./self.sigma0)
        finf = problemDC.fields(1./self.sigmaInf)
        datainf = self.surveyDC.dpred(1./self.sigmaInf, f=finf)
        problemIP = IP.Problem2D_CC(
            self.mesh,
            rho=1./self.sigmaInf,
            etaMap=Maps.IdentityMap(self.mesh)
        )
        problemIP.Solver = Solver
        surveyIP = IP.Survey(self.srcLists_ip)
        problemIP.pair(surveyIP)
        data_full = data0 - datainf
        data = surveyIP.dpred(self.eta)
        err = np.linalg.norm((data-data_full)/data_full)**2 / data_full.size
        if err < 0.05:
            passed = True
            print(">> IP forward test for Problem2D_CC is passed")
        else:
            import matplotlib.pyplot as plt
            passed = False
            print(">> IP forward test for Problem2D_CC is failed")
            print(err)
            plt.plot(data_full)
            plt.plot(data, 'k.')
            plt.show()

        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
