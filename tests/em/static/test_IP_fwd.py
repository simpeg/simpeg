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
        npad = 2
        hx = [(cs, npad, -1.3), (cs, 21), (cs, npad, 1.3)]
        hy = [(cs, npad, -1.3), (cs, 21), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, 20)]
        mesh = Mesh.TensorMesh([hx, hy, hz], x0="CCN")

        x = mesh.vectorCCx[(mesh.vectorCCx > -80.) & (mesh.vectorCCx < 80.)]
        y = mesh.vectorCCy[(mesh.vectorCCy > -80.) & (mesh.vectorCCy < 80.)]
        Aloc = np.r_[-100., 0., 0.]
        Bloc = np.r_[100., 0., 0.]
        M = Utils.ndgrid(x-12.5, y, np.r_[0.])
        N = Utils.ndgrid(x+12.5, y, np.r_[0.])
        radius = 50.
        xc = np.r_[0., 0., -100]
        blkind = Utils.ModelBuilder.getIndicesSphere(xc, radius, mesh.gridCC)
        sigmaInf = np.ones(mesh.nC)*1e-2
        eta = np.zeros(mesh.nC)
        eta[blkind] = 0.1
        sigma0 = sigmaInf*(1.-eta)

        rx = DC.Rx.Dipole(M, N)
        src = DC.Src.Dipole([rx], Aloc, Bloc)
        surveyDC = DC.Survey([src])

        self.surveyDC = surveyDC
        self.mesh = mesh
        self.sigmaInf = sigmaInf
        self.sigma0 = sigma0
        self.src = src
        self.eta = eta

    def test_Problem3D_N(self):

        problemDC = DC.Problem3D_N(
            self.mesh, sigmaMap=Maps.IdentityMap(self.mesh)
        )
        problemDC.Solver = Solver
        problemDC.pair(self.surveyDC)
        data0 = self.surveyDC.dpred(self.sigma0)
        finf = problemDC.fields(self.sigmaInf)
        datainf = self.surveyDC.dpred(self.sigmaInf, f=finf)
        problemIP = IP.Problem3D_N(
            self.mesh,
            sigma=self.sigmaInf,
            etaMap=Maps.IdentityMap(self.mesh),
            Ainv=problemDC.Ainv,
            _f=finf
        )
        problemIP.Solver = Solver
        surveyIP = IP.Survey([self.src])
        problemIP.pair(surveyIP)
        data_full = data0 - datainf
        data = surveyIP.dpred(self.eta)
        err = np.linalg.norm((data-data_full)/data_full)**2 / data_full.size
        if err < 0.05:
            passed = True
            print(">> IP forward test for Problem3D_N is passed")
        else:
            passed = False
            print(">> IP forward test for Problem3D_N is failed")
        self.assertTrue(passed)

    def test_Problem3D_CC(self):

        problemDC = DC.Problem3D_CC(
            self.mesh, sigmaMap=Maps.IdentityMap(self.mesh)
        )
        problemDC.Solver = Solver
        problemDC.pair(self.surveyDC)
        data0 = self.surveyDC.dpred(self.sigma0)
        finf = problemDC.fields(self.sigmaInf)
        datainf = self.surveyDC.dpred(self.sigmaInf, f=finf)
        problemIP = IP.Problem3D_CC(
            self.mesh,
            rho=1./self.sigmaInf,
            etaMap=Maps.IdentityMap(self.mesh),
            Ainv=problemDC.Ainv,
            _f=finf
        )
        problemIP.Solver = Solver
        surveyIP = IP.Survey([self.src])
        problemIP.pair(surveyIP)
        data_full = data0 - datainf
        data = surveyIP.dpred(self.eta)
        err = np.linalg.norm((data-data_full)/data_full)**2 / data_full.size
        if err < 0.05:
            passed = True
            print(">> IP forward test for Problem3D_CC is passed")
        else:
            passed = False
            print(">> IP forward test for Problem3D_CC is failed")
        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
