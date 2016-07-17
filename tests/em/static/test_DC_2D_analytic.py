from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest
from SimPEG import Mesh, Utils, EM, Maps, np
import SimPEG.EM.Static.DC as DC

class DCProblemAnalyticTests(unittest.TestCase):

    def setUp(self):

        cs = 12.5
        hx = [(cs,7, -1.3),(cs,61),(cs,7, 1.3)]
        hy = [(cs,7, -1.3),(cs,20)]
        mesh = Mesh.TensorMesh([hx, hy],x0="CN")
        sighalf = 1e-2
        sigma = np.ones(mesh.nC)*sighalf
        x = np.linspace(-135, 250., 20)
        M = Utils.ndgrid(x-12.5, np.r_[0.])
        N = Utils.ndgrid(x+12.5, np.r_[0.])
        A0loc = np.r_[-150, 0.]
        A1loc = np.r_[-130, 0.]
        rxloc = [np.c_[M, np.zeros(20)], np.c_[N, np.zeros(20)]]
        data_anal = EM.Analytics.DCAnalyticHalf(np.r_[A0loc, 0.], rxloc, sighalf, earth_type="halfspace")

        rx = DC.Rx.Dipole_ky(M, N)
        src0 = DC.Src.Pole([rx], A0loc)
        survey = DC.Survey_ky([src0])

        self.survey = survey
        self.mesh = mesh
        self.sigma = sigma
        self.data_anal = data_anal

        try:
            from pymatsolver import MumpsSolver
            self.Solver = MumpsSolver
        except ImportError as e:
            self.Solver = SolverLU

    def test_Problem3D_N(self):

        problem = DC.Problem2D_N(self.mesh)
        problem.Solver = self.Solver
        problem.pair(self.survey)
        data = self.survey.dpred(self.sigma)
        err= old_div(np.linalg.norm(old_div((data-self.data_anal),self.data_anal))**2, self.data_anal.size)
        if err < 0.05:
            passed = True
            print(">> DC analytic test for Problem3D_N is passed")
        else:
            passed = False
            print(">> DC analytic test for Problem3D_N is failed")
        self.assertTrue(passed)

    def test_Problem3D_CC(self):
        problem = DC.Problem2D_CC(self.mesh)
        problem.Solver = self.Solver
        problem.pair(self.survey)
        data = self.survey.dpred(self.sigma)
        err= old_div(np.linalg.norm(old_div((data-self.data_anal),self.data_anal))**2, self.data_anal.size)
        if err < 0.05:
            passed = True
            print(">> DC analytic test for Problem3D_CC is passed")
        else:
            passed = False
            print(">> DC analytic test for Problem3D_CC is failed")
        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()

