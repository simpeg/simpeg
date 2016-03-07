import unittest
from SimPEG import *
from SimPEG import EM

plotIt = False
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
            A = prb.getA(tInd)
            Av = A*v
            prb.curModel = m0 
            ADeriv_dm = lambda dm: prb.getADeriv(tInd, v, dm)

            return Av, ADeriv_dm

        Tests.checkDerivative(AderivTest, m0, plotIt=False, num=4, eps=1e-20)   


    def JvecTest(self, rxcomp): 
        prb, m, mesh = setUp(rxcomp)

        derChk = lambda m: [prb.survey.dpred(m), lambda mx: prb.Jvec(m, mx)]
        print '\n'
        print 'test_Jvec_%s' %(rxcomp)
        Tests.checkDerivative(derChk, m, plotIt=False, num=2, eps=1e-20)

    def test_Jvec_b_bx(self):
        self.JvecTest('bx')

    def test_Jvec_b_bz(self):
        self.JvecTest('bz')

    def test_Jvec_b_ey(self):
        self.JvecTest('ey')
        

    # def test_projectAdjoint(self):
    #     prb = self.prb
    #     survey = prb.survey
    #     mesh = self.mesh

    #     # Generate random fields and data
    #     f = EM.TDEM.FieldsTDEM(prb.mesh, prb.survey)
    #     for i in range(prb.nT):
    #         f[:,'b',i] = np.random.rand(mesh.nF, 1)
    #         f[:,'e',i] = np.random.rand(mesh.nE, 1)
    #     d_vec = np.random.rand(survey.nD)
    #     d = Survey.Data(survey,v=d_vec)
 
    #     # Check that d.T*Q*f = f.T*Q.T*d
    #     V1 = d_vec.dot(survey.evalDeriv(None, v=f).tovec())
    #     V2 = f.tovec().dot(survey.evalDeriv(None, v=d, adjoint=True).tovec())

    #     self.assertTrue((V1-V2)/np.abs(V1) < tol)

    # def test_adjointAhVsAht(self):
    #     prb = self.prb
    #     mesh = self.mesh
    #     sigma = self.sigma

    #     f1 = EM.TDEM.FieldsTDEM(prb.mesh, prb.survey)
    #     for i in range(1,prb.nT+1):
    #         f1[:,'b',i] = np.random.rand(mesh.nF, 1)
    #         f1[:,'e',i] = np.random.rand(mesh.nE, 1)

    #     f2 = EM.TDEM.FieldsTDEM(prb.mesh, prb.survey)
    #     for i in range(1,prb.nT+1):
    #         f2[:,'b',i] = np.random.rand(mesh.nF, 1)
    #         f2[:,'e',i] = np.random.rand(mesh.nE, 1)

    #     V1 = f2.tovec().dot(prb._AhVec(sigma, f1).tovec())
    #     V2 = f1.tovec().dot(prb._AhtVec(sigma, f2).tovec())
    #     self.assertTrue(np.abs(V1-V2)/np.abs(V1) < tol)

    # # def test_solveAhtVsAhtVec(self):
    # #     prb = self.prb
    # #     mesh = self.mesh
    # #     sigma = np.random.rand(prb.mapping.nP)

    # #     f1 = EM.TDEM.FieldsTDEM(mesh,prb.survey)
    # #     for i in range(1,prb.nT+1):
    # #         f1[:,'b',i] = np.random.rand(mesh.nF, 1)
    # #         f1[:,'e',i] = np.random.rand(mesh.nE, 1)

    # #     f2 = prb.solveAht(sigma, f1)
    # #     f3 = prb._AhtVec(sigma, f2)

    # #     if True:
    # #         import matplotlib.pyplot as plt
    # #         plt.plot(f3.tovec(),'b')
    # #         plt.plot(f1.tovec(),'r')
    # #         plt.show()
    # #     V1 = np.linalg.norm(f3.tovec()-f1.tovec())
    # #     V2 = np.linalg.norm(f1.tovec())
    # #     print 'AhtVsAhtVec', V1, V2, f1.tovec()
    # #     print 'I am gunna fail this one: boo. :('
    # #     self.assertLess(V1/V2, 1e-6)

    # # def test_adjointsolveAhVssolveAht(self):
    # #     prb = self.prb
    # #     mesh = self.mesh
    # #     sigma = self.sigma

    # #     f1 = EM.TDEM.FieldsTDEM(prb.mesh, prb.survey)
    # #     for i in range(1,prb.nT+1):
    # #         f1[:,'b',i] = np.random.rand(mesh.nF, 1)
    # #         f1[:,'e',i] = np.random.rand(mesh.nE, 1)

    # #     f2 = EM.TDEM.FieldsTDEM(prb.mesh, prb.survey)
    # #     for i in range(1,prb.nT+1):
    # #         f2[:,'b',i] = np.random.rand(mesh.nF, 1)
    # #         f2[:,'e',i] = np.random.rand(mesh.nE, 1)

    # #     V1 = f2.tovec().dot(prb.solveAh(sigma, f1).tovec())
    # #     V2 = f1.tovec().dot(prb.solveAht(sigma, f2).tovec())
    # #     print V1, V2
    # #     self.assertLess(np.abs(V1-V2)/np.abs(V1), 1e-6)

    # def test_adjointGvecVsGtvec(self):
    #     mesh = self.mesh
    #     prb = self.prb

    #     m = np.random.rand(prb.mapping.nP)
    #     sigma = np.random.rand(prb.mapping.nP)

    #     u = EM.TDEM.FieldsTDEM(prb.mesh, prb.survey)
    #     for i in range(1,prb.nT+1):
    #         u[:,'b',i] = np.random.rand(mesh.nF, 1)
    #         u[:,'e',i] = np.random.rand(mesh.nE, 1)

    #     v = EM.TDEM.FieldsTDEM(prb.mesh, prb.survey)
    #     for i in range(1,prb.nT+1):
    #         v[:,'b',i] = np.random.rand(mesh.nF, 1)
    #         v[:,'e',i] = np.random.rand(mesh.nE, 1)

    #     V1 = m.dot(prb.Gtvec(sigma, v, u))
    #     V2 = v.tovec().dot(prb.Gvec(sigma, m, u).tovec())
    #     self.assertTrue(np.abs(V1-V2)/np.abs(V1) < tol)

    # def test_adjointJvecVsJtvec(self):
    #     mesh = self.mesh
    #     prb = self.prb
    #     sigma = self.sigma

    #     m = np.random.rand(prb.mapping.nP)
    #     d = np.random.rand(prb.survey.nD)

    #     V1 = d.dot(prb.Jvec(sigma, m))
    #     V2 = m.dot(prb.Jtvec(sigma, d))
    #     passed = np.abs(V1-V2)/np.abs(V1) < tol
    #     print 'AdjointTest', V1, V2, passed
    #     self.assertTrue(passed)




if __name__ == '__main__':
    unittest.main()
