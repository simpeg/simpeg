import unittest
from SimPEG import *
from SimPEG import EM

plotIt = False

testDeriv   = True
testAdjoint = False

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

    try:
        from pymatsolver import MumpsSolver
        prb.Solver = MumpsSolver
    except ImportError, e:
        prb.Solver = SolverLU

    m = np.log(1e-1)*np.ones(prb.mapping.nP) + 1e-2*np.random.randn(prb.mapping.nP)

    prb.pair(survey)
    mesh = mesh

    return prb, m, mesh

class TDEM_bDerivTests(unittest.TestCase):

    

    def test_ADeriv(self):
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

        Tests.checkDerivative(AderivTest, m0, plotIt=False, num=4, eps=1e-20)   


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
    

    if testAdjoint: 
        def test_adjointJvecVsJtvec(self):
            prb, m0, mesh = setUp()

            m = np.random.rand(prb.mapping.nP)
            d = np.random.rand(prb.survey.nD)

            V1 = d.dot(prb.Jvec(m0, m))
            V2 = m.dot(prb.Jtvec(m0, d))
            passed = np.abs(V1-V2)/np.abs(V1) < tol
            print 'AdjointTest', V1, V2, passed
            self.assertTrue(passed)




if __name__ == '__main__':
    unittest.main()
