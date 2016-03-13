import unittest
from SimPEG import *
from SimPEG import EM

plotIt = False

testDeriv   = False
testAdjoint = True

tol = 1e-6

def setUp(rxcomp='bz'):
    cs = 5.
    ncx = 20
    ncy = 10
    npad = 20
    hx = [(cs,ncx), (cs,npad,1.3)]
    hy = [(cs,npad,-1.3), (cs,ncy), (cs,npad,1.3)]
    mesh = Mesh.CylMesh([hx,1,hy], '00C')
#
    active = mesh.vectorCCz<0.
    activeMap = Maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
    mapping = Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * activeMap

    rxOffset = 10.
    rx = EM.TDEM.Rx(np.array([[rxOffset, 0., -1e-2]]), np.logspace(-4,-3, 20), rxcomp)
    src = EM.TDEM.SurveyTDEM.MagDipole([rx], loc=np.array([0., 0., 0.]))

    survey = EM.TDEM.Survey([src])

    prb = EM.TDEM.Problem_b(mesh, mapping=mapping)
    prb.timeSteps = [(1e-05, 10), (5e-05, 10), (2.5e-4, 10)]
    # prb.timeSteps = [(1e-05, 10), (1e-05, 50), (1e-05, 50) ] #, (2.5e-4, 10)]

    try:
        from pymatsolver import MumpsSolver
        prb.Solver = MumpsSolver
    except ImportError, e:
        prb.Solver = SolverLU

    m = np.log(1e-1)*np.ones(prb.mapping.nP) + 1e-2*np.random.randn(prb.mapping.nP)

    prb.pair(survey)
    mesh = mesh

    return prb, m, mesh

class TDEM_DerivTests(unittest.TestCase):

# ====== TEST A ========== #

    def test_Deriv_Pieces(self):
        prb, m0, mesh = setUp()
        tInd = 0

        v = np.random.rand(mesh.nF)

        def AderivTest(m):
            prb.curModel = m
            A = prb.getAdiag(tInd)
            Av = A*v
            prb.curModel = m0
            ADeriv_dm = lambda dm: prb.getAdiagDeriv(tInd, v, dm)

            return Av, ADeriv_dm

        def A_adjointTest():
            print '\n Testing A_adjoint'
            m = np.random.rand(prb.mapping.nP)
            v = np.random.rand(prb.mesh.nF)
            u = np.random.rand(prb.mesh.nF)
            prb.curModel = m0

            tInd = 0 # not actually used
            V1 = v.dot(prb.getAdiagDeriv(tInd, u, m))
            V2 = m.dot(prb.getAdiagDeriv(tInd, u, v, adjoint=True))
            passed = np.abs(V1-V2)/np.abs(V1) < tol
            print 'AdjointTest', V1, V2, passed
            self.assertTrue(passed)

        print '\n Testing ADeriv'
        Tests.checkDerivative(AderivTest, m0, plotIt=False, num=4, eps=1e-20)
        A_adjointTest()


# ====== TEST Jvec ========== #

    def JvecTest(self, rxcomp):
        prb, m, mesh = setUp(rxcomp)

        derChk = lambda m: [prb.survey.dpred(m), lambda mx: prb.Jvec(m, mx)]
        print '\n'
        print 'test_Jvec_%s' %(rxcomp)
        Tests.checkDerivative(derChk, m, plotIt=False, num=2, eps=1e-20)

    if testDeriv:
        def test_Jvec_b_bx(self):
            self.JvecTest('bx')

        def test_Jvec_b_bz(self):
            self.JvecTest('bz')

        def test_Jvec_b_ey(self):
            self.JvecTest('ey')


# ====== TEST Jtvec ========== #

    def adjointJvecVsJtvecTest(self, rxcomp='bz'):
        print '\n Adjoint Testing Jvec, Jtvec %s' %(rxcomp)
        prb, m0, mesh = setUp(rxcomp)

        m = np.random.rand(prb.mapping.nP)
        d = np.random.randn(prb.survey.nD)

        V1 = d.dot(prb.Jvec(m0, m))
        V2 = m.dot(prb.Jtvec(m0, d))
        passed = np.abs(V1-V2)/np.abs(V1) < tol
        print 'AdjointTest', V1, V2, passed
        self.assertTrue(passed)

    if testAdjoint:
        def test_Jvec_b_bx(self):
            self.adjointJvecVsJtvecTest('bx')

        def test_Jvec_b_bz(self):
            self.adjointJvecVsJtvecTest('bz')

        def test_Jvec_b_ey(self):
            self.adjointJvecVsJtvecTest('ey')


if __name__ == '__main__':
    unittest.main()
