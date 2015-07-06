import unittest
from SimPEG import *
import simpegEM as EM

plotIt = False

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
        activeMap = Maps.ActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
        mapping = Maps.ExpMap(mesh) * Maps.Vertical1DMap(mesh) * activeMap

        rxOffset = 40.
        rx = EM.TDEM.RxTDEM(np.array([[rxOffset, 0., 0.]]), np.logspace(-4,-3, 20), 'bz')
        src = EM.TDEM.SrcTDEM_VMD_MVP( [rx], loc=np.array([0., 0., 0.]))
        rx2 = EM.TDEM.RxTDEM(np.array([[rxOffset-10, 0., 0.]]), np.logspace(-5,-4, 25), 'bz')
        src2 = EM.TDEM.SrcTDEM_VMD_MVP( [rx2], loc=np.array([0., 0., 0.]))

        survey = EM.TDEM.SurveyTDEM([src,src2])

        self.prb = EM.TDEM.ProblemTDEM_b(mesh, mapping=mapping)
        # self.prb.timeSteps = [1e-5]
        self.prb.timeSteps = [(1e-05, 10), (5e-05, 10), (2.5e-4, 10)]
        # self.prb.timeSteps = [(1e-05, 100)]

        try:
            from pymatsolver import MumpsSolver
            self.prb.Solver = MumpsSolver
        except ImportError, e:
            self.prb.Solver  = SolverLU

        self.sigma = np.ones(mesh.nCz)*1e-8
        self.sigma[mesh.vectorCCz<0] = 1e-1
        self.sigma = np.log(self.sigma[active])

        self.prb.pair(survey)
        self.mesh = mesh

    def test_DerivG(self):
        """
            Test the derivative of c with respect to sigma
        """

        # Random model and perturbation
        sigma = np.random.rand(self.prb.mapping.nP)

        f = self.prb.fields(sigma)
        dm = 1000*np.random.rand(self.prb.mapping.nP)
        h = 0.01

        derChk = lambda m: [self.prb._AhVec(m, f).tovec(), lambda mx: self.prb.Gvec(sigma, mx, u=f).tovec()]
        print '\ntest_DerivG'
        Tests.checkDerivative(derChk, sigma, plotIt=False, dx=dm, num=4, eps=1e-20)

    def test_Deriv_dUdM(self):

        prb = self.prb
        prb.timeSteps = [(1e-05, 10), (0.0001, 10), (0.001, 10)]
        mesh = self.mesh
        sigma = self.sigma

        dm = 10*np.random.rand(prb.mapping.nP)
        f = prb.fields(sigma)

        derChk = lambda m: [self.prb.fields(m).tovec(), lambda mx: -prb.solveAh(sigma, prb.Gvec(sigma, mx, u=f)).tovec()]
        print '\n'
        print 'test_Deriv_dUdM'
        Tests.checkDerivative(derChk, sigma, plotIt=False, dx=dm, num=4, eps=1e-20)

    def test_Deriv_J(self):

        prb = self.prb
        prb.timeSteps = [(1e-05, 10), (0.0001, 10), (0.001, 10)]
        mesh = self.mesh
        sigma = self.sigma

        # d_sig = 0.8*sigma #np.random.rand(mesh.nCz)
        d_sig = 10*np.random.rand(prb.mapping.nP)


        derChk = lambda m: [prb.survey.dpred(m), lambda mx: prb.Jvec(sigma, mx)]
        print '\n'
        print 'test_Deriv_J'
        Tests.checkDerivative(derChk, sigma, plotIt=False, dx=d_sig, num=4, eps=1e-20)

    def test_projectAdjoint(self):
        prb = self.prb
        survey = prb.survey
        nSrc = survey.nSrc
        mesh = self.mesh

        # Generate random fields and data
        f = EM.TDEM.FieldsTDEM(prb.mesh, prb.survey)
        for i in range(prb.nT):
            f[:,'b',i] = np.random.rand(mesh.nF, nSrc)
            f[:,'e',i] = np.random.rand(mesh.nE, nSrc)
        d_vec = np.random.rand(survey.nD)
        d = Survey.Data(survey,v=d_vec)

        # Check that d.T*Q*f = f.T*Q.T*d
        V1 = d_vec.dot(survey.projectFieldsDeriv(None, v=f).tovec())
        V2 = np.sum((f.tovec())*(survey.projectFieldsDeriv(None, v=d, adjoint=True).tovec()))

        self.assertTrue((V1-V2)/np.abs(V1) < 1e-6)

    def test_adjointGvecVsGtvec(self):
        mesh = self.mesh
        prb = self.prb

        m = np.random.rand(prb.mapping.nP)
        sigma = np.random.rand(prb.mapping.nP)

        u = EM.TDEM.FieldsTDEM(prb.mesh, prb.survey)
        for i in range(1,prb.nT+1):
            u[:,'b',i] = np.random.rand(mesh.nF, 2)
            u[:,'e',i] = np.random.rand(mesh.nE, 2)

        v = EM.TDEM.FieldsTDEM(prb.mesh, prb.survey)
        for i in range(1,prb.nT+1):
            v[:,'b',i] = np.random.rand(mesh.nF, 2)
            v[:,'e',i] = np.random.rand(mesh.nE, 2)

        V1 = m.dot(prb.Gtvec(sigma, v, u))
        V2 = np.sum(v.tovec()*prb.Gvec(sigma, m, u).tovec())
        self.assertTrue(np.abs(V1-V2)/np.abs(V1) <1e-6)

    def test_adjointJvecVsJtvec(self):
        mesh = self.mesh
        prb = self.prb
        sigma = self.sigma

        m = np.random.rand(prb.mapping.nP)
        d = np.random.rand(prb.survey.nD)

        V1 = d.dot(prb.Jvec(sigma, m))
        V2 = m.dot(prb.Jtvec(sigma, d))
        print 'AdjointTest', V1, V2
        self.assertTrue(np.abs(V1-V2)/np.abs(V1) < 1e-6)



if __name__ == '__main__':
    unittest.main()
