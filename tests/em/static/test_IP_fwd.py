import unittest
from SimPEG import Mesh, Utils, EM, Maps, np
import SimPEG.EM.Static.DC as DC

class IPProblemAnalyticTests(unittest.TestCase):

    def setUp(self):

        cs = 12.5
        hx = [(cs,2, -1.3),(cs,61),(cs,2, 1.3)]
        hy = [(cs,2, -1.3),(cs,20)]
        mesh = Mesh.TensorMesh([hx, hy],x0="CN")
        sighalf = 1e-2
        sigma = np.ones(mesh.nC)*sighalf
        x = np.linspace(-135, 250., 20)
        M = Utils.ndgrid(x-12.5, np.r_[0.])
        N = Utils.ndgrid(x+12.5, np.r_[0.])
        A0loc = np.r_[-150, 0.]
        A1loc = np.r_[-130, 0.]
        rxloc = [np.c_[M, np.zeros(20)], np.c_[N, np.zeros(20)]]

        blkind = Utils.ModelBuilder.getIndicesSphere(xc, radius, mesh.gridCC)
        sigmaInf = np.ones(mesh.nC)*1e-2
        eta = np.zeros(mesh.nC)
        eta[blkind] = 0.1
        sigma0 = sigmaInf*(1.-eta)

        rx = DC.Rx.Dipole_ky(M, N)
        src0 = DC.Src.Pole([rx], A0loc)
        surveyDC = DC.Survey([src0])
        surveyIP = DC.Survey_ky([src0])

        self.surveyDC = surveyDC
        self.surveyIP = surveyIP

        self.mesh = mesh
        self.sigma = sigma
        self.data_anal = data_anal

        try:
            from pymatsolver import MumpsSolver
            self.Solver = MumpsSolver
        except ImportError, e:
            self.Solver = SolverLU

    def test_Problem3D_N(self):

        problem = DC.Problem2D_N(self.mesh)
        problem.Solver = self.Solver
        problem.pair(self.survey)
        data = self.survey.dpred(self.sigma)
        err= np.linalg.norm((data-self.data_anal)/self.data_anal)**2 / self.data_anal.size
        if err < 0.05:
            passed = True
            print ">> DC analytic test for Problem3D_N is passed"
        else:
            passed = False
            print ">> DC analytic test for Problem3D_N is failed"
        self.assertTrue(passed)

    def test_Problem3D_CC(self):
        problem = DC.Problem2D_CC(self.mesh)
        problem.Solver = self.Solver
        problem.pair(self.survey)
        data = self.survey.dpred(self.sigma)
        err= np.linalg.norm((data-self.data_anal)/self.data_anal)**2 / self.data_anal.size
        if err < 0.05:
            passed = True
            print ">> DC analytic test for Problem3D_CC is passed"
        else:
            passed = False
            print ">> DC analytic test for Problem3D_CC is failed"
        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()

