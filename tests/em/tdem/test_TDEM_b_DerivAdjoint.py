import unittest
from SimPEG import *
from SimPEG import EM

plotIt = False
tol = 1e-6

class TDEM_bDerivTests(unittest.TestCase):

    def setUp(self):

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
        rx = EM.TDEM.Rx(np.array([[rxOffset, 0., -1e-2]]), np.logspace(-4,-3, 20), 'bz')
        src = EM.TDEM.SurveyTDEM.MagDipole([rx], loc=np.array([0., 0., 0.]))

        survey = EM.TDEM.Survey([src])

        self.prb = EM.TDEM.Problem_b(mesh, mapping=mapping)
        # self.prb.timeSteps = [1e-5]
        self.prb.timeSteps = [(1e-05, 10), (5e-05, 10), (2.5e-4, 10)]
        # self.prb.__makeASymmetric = False
        # self.prb.timeSteps = [(1e-05, 100)]

        try:
            from pymatsolver import MumpsSolver
            self.prb.Solver = MumpsSolver
        except ImportError, e:
            self.prb.Solver = SolverLU

        # self.sigma = np.ones(mesh.nCz)*1e-8
        # self.sigma[active] = 1e-1 
        # self.sigma[active] += 1e-2*np.random.rand(len(active))
        self.m = np.log(1e-1)*np.ones(self.prb.mapping.nP) + 1e-2*np.random.randn(self.prb.mapping.nP)

        self.prb.pair(survey)
        self.mesh = mesh

    # def test_AhVec(self):
    #     """
    #         Test that fields and AhVec produce consistent results
    #     """

    #     prb = self.prb
    #     sigma = self.sigma

    #     u = prb.fields(sigma)
    #     Ahu = prb._AhVec(sigma, u)

    #     V1 = Ahu[:,'b',1]
    #     V2 = 1./prb.timeSteps[0]*prb.MfMui*u[:,'b',0]
    #     self.assertLess(np.linalg.norm(V1-V2)/np.linalg.norm(V2), 1.e-6)

    #     V1 = Ahu[:,'e',1]
    #     return np.linalg.norm(V1) < 1.e-6

    #     for i in range(2,prb.nT):

    #         dt = prb.timeSteps[i]

    #         V1 = Ahu[:,'b',i]
    #         V2 = 1.0/dt*prb.MfMui*u[:,'b', i-1]
    #         # print np.linalg.norm(V1), np.linalg.norm(V2)
    #         self.assertLess(np.linalg.norm(V1)/np.linalg.norm(V2), 1.e-6)

    #         V1 = Ahu[:,'e',i]
    #         V2 = prb.MeSigma*u[:,'e',i]
    #         # print np.linalg.norm(V1), np.linalg.norm(V2)
    #         return np.linalg.norm(V1)/np.linalg.norm(V2), 1.e-6

    # def test_AhVecVSMat_OneTS(self):

    #     prb = self.prb
    #     prb.timeSteps = [1e-05]
    #     sigma = self.sigma
    #     prb.curModel = sigma

    #     dt = prb.timeSteps[0]
    #     a11 = 1/dt*prb.MfMui*sp.identity(prb.mesh.nF)
    #     a12 = prb.MfMui*prb.mesh.edgeCurl
    #     a21 = prb.mesh.edgeCurl.T*prb.MfMui
    #     a22 = -prb.MeSigma
    #     A = sp.bmat([[a11,a12],[a21,a22]])

    #     f = prb.fields(sigma)
    #     u1 = A*f.tovec()
    #     u2 = prb._AhVec(sigma,f).tovec()

    #     self.assertTrue(np.linalg.norm(u1-u2)/np.linalg.norm(u1)<1e-12)

    # def test_solveAhVSMat_OneTS(self):
    #     prb = self.prb

    #     prb.timeSteps = [1e-05]

    #     sigma = self.sigma
    #     prb.curModel = sigma

    #     dt = prb.timeSteps[0]
    #     a11 = 1.0/dt*prb.MfMui*sp.identity(prb.mesh.nF)
    #     a12 = prb.MfMui*prb.mesh.edgeCurl
    #     a21 = prb.mesh.edgeCurl.T*prb.MfMui
    #     a22 = -prb.MeSigma
    #     A = sp.bmat([[a11,a12],[a21,a22]])

    #     f = prb.fields(sigma)
    #     f[:,:,0] = {'b':0}
    #     f[:,'b',1] = 0

    #     self.assertTrue(np.all(np.r_[f[:,'b',1],f[:,'e',1]] == f.tovec()))

    #     u1 = prb.solveAh(sigma,f).tovec().flatten()
    #     u2 = sp.linalg.spsolve(A.tocsr(),f.tovec())

    #     self.assertTrue(np.linalg.norm(u1-u2)<1e-8)

    # def test_solveAhVsAhVec(self):

    #     prb = self.prb
    #     mesh = self.prb.mesh
    #     sigma = self.sigma
    #     self.prb.curModel = sigma

    #     f = EM.TDEM.FieldsTDEM(prb.mesh, prb.survey)
    #     f[:,'b',:] = 0.0
    #     for i in range(prb.nT):
    #         f[:,'e', i] = np.random.rand(mesh.nE, 1)

    #     Ahf = prb._AhVec(sigma, f)
    #     f_test = prb.solveAh(sigma, Ahf)

    #     u1 = f.tovec()
    #     u2 = f_test.tovec()
    #     self.assertTrue(np.linalg.norm(u1-u2)<1e-8)

    # def test_DerivG(self):
    #     """
    #         Test the derivative of c with respect to sigma
    #     """

    #     # Random model and perturbation
    #     sigma = np.random.rand(self.prb.mapping.nP)

    #     f = self.prb.fields(sigma)
    #     dm = 1000*np.random.rand(self.prb.mapping.nP)
    #     h = 0.01

    #     derChk = lambda m: [self.prb._AhVec(m, f).tovec(), lambda mx: self.prb.Gvec(sigma, mx, u=f).tovec()]
    #     print '\ntest_DerivG'
    #     passed = Tests.checkDerivative(derChk, sigma, plotIt=False, dx=dm, num=4, eps=1e-20)
    #     return passed

    # def test_Deriv_dUdM(self):

    #     prb = self.prb
    #     prb.timeSteps = [(1e-05, 10), (0.0001, 10), (0.001, 10)]
    #     mesh = self.mesh
    #     sigma = self.sigma

    #     dm = 10*np.random.rand(prb.mapping.nP)
    #     f = prb.fields(sigma)

    #     derChk = lambda m: [self.prb.fields(m).tovec(), lambda mx: -prb.solveAh(sigma, prb.Gvec(sigma, mx, u=f)).tovec()]
    #     print '\n'
    #     print 'test_Deriv_dUdM'
    #     Tests.checkDerivative(derChk, sigma, plotIt=False, dx=dm, num=4, eps=1e-20)

    def test_ADeriv(self):
        prb = self.prb
        tInd = 0

        v = np.random.rand(self.mesh.nF)

        def AderivTest(m):
            prb.curModel = m
            A = prb.getA(tInd)
            Av = A*v
            prb.curModel = self.m 
            ADeriv_dm = lambda dm: prb.getADeriv(tInd, v, dm)

            return Av, ADeriv_dm

        Tests.checkDerivative(AderivTest, self.m, plotIt=False, num=4, eps=1e-20)   

    # def test_Fields_Deriv(self):
    #     prb = self.prb
    #     tInd = 10

    #     v = np.random.rand(self.mesh.nF)

    #     def FieldsDerivs(m):

    #         sol = prb.fields(m)[:,'bSolution',tInd]

    #         prb.curModel = self.m
    #         f = prb.fields(self.m)

    #         df_dm_v = EM.TDEM.FieldsTDEM.Fields_Derivs(mesh, survey)
    #         for i in range(tInd):
    #             Ainv = prb.Solver(prb.getA(tInd))
    #             df_dm_v = Ainv * 

    #         deriv = lambda dm: f._bDeriv(prb.survey.srcList[0], f[:,'bSolution',tInd], dm)

    #         return sol, deriv

    #     Tests.checkDerivative(FieldsDerivs, self.m, plotIt=False, num=4, eps=1e-20) 

    def test_Deriv_J(self):

        prb = self.prb
        # prb.timeSteps = [(1e-05, 10), (0.0001, 10), (0.001, 10)]
        mesh = self.mesh
        m = self.m

        # d_sig = 0.8*sigma #np.random.rand(mesh.nCz)
        # d_m = 0.1*np.random.randn(prb.mapping.nP)


        derChk = lambda m: [prb.survey.dpred(m), lambda mx: prb.Jvec(m, mx)]
        print '\n'
        print 'test_Deriv_J'
        Tests.checkDerivative(derChk, m, plotIt=False, num=2, eps=1e-20)

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
