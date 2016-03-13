import unittest
from SimPEG import *
from SimPEG import EM

plotIt = False
testDeriv = True
testAdjoint = True

TOL = 1e-6

class TDEM_bDerivTests(unittest.TestCase):

    def setUp(self):

        cs = 5.
        ncx = 20
        ncy = 6
        npad = 20
        hx = [(cs,ncx), (cs,npad,1.3)]
        hy = [(cs,npad,-1.3), (cs,ncy), (cs,npad,1.3)]
        mesh = Mesh.CylMesh([hx,1,hy], '00C')

        active = mesh.vectorCCz<0.
        activeMap = Maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
        mapping = Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * activeMap

        rxOffset = 40.
        rx = EM.TDEM.Rx(np.array([[rxOffset, 0., 0.]]), np.logspace(-4,-3, 20), 'bz')
        src = EM.TDEM.SurveyTDEM.MagDipole( [rx], loc=np.array([0., 0., 0.]))
        rx2 = EM.TDEM.Rx(np.array([[rxOffset-10, 0., 0.]]), np.logspace(-5,-4, 25), 'bz')
        src2 = EM.TDEM.SurveyTDEM.MagDipole( [rx2], loc=np.array([0., 0., 0.]))

        survey = EM.TDEM.Survey([src,src2])

        self.prb = EM.TDEM.Problem_b(mesh, mapping=mapping)
        # self.prb.timeSteps = [1e-5]
        self.prb.timeSteps = [(1e-05, 10), (5e-05, 10), (2.5e-4, 10)]
        # self.prb.timeSteps = [(1e-05, 100)]

        try:
            from pymatsolver import MumpsSolver
            self.prb.Solver = MumpsSolver
        except ImportError, e:
            self.prb.Solver  = SolverLU

        self.m = np.log(1e-1)*np.ones(self.prb.mapping.nP) + 1e-2*np.random.randn(self.prb.mapping.nP)

        self.prb.pair(survey)
        self.mesh = mesh

    if testDeriv:
        def test_Deriv_J(self):

            prb = self.prb
            prb.timeSteps = [(1e-05, 10), (0.0001, 10), (0.001, 10)]
            mesh = self.mesh

            derChk = lambda m: [prb.survey.dpred(m), lambda mx: prb.Jvec(self.m, mx)]
            print '\n'
            print 'test_Deriv_J'
            Tests.checkDerivative(derChk, self.m, plotIt=False, num=3, eps=1e-20)

    if testAdjoint:
        def test_adjointJvecVsJtvec(self):
            mesh = self.mesh
            prb = self.prb
            m0 = self.m

            m = np.random.rand(prb.mapping.nP)
            d = np.random.rand(prb.survey.nD)

            V1 = d.dot(prb.Jvec(m0, m))
            V2 = m.dot(prb.Jtvec(m0, d))

            tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.
            passed = np.abs(V1-V2) < tol
            print '    ', V1, V2, np.abs(V1-V2), tol, passed
            self.assertTrue(passed)



if __name__ == '__main__':
    unittest.main()
