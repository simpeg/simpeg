import unittest
from SimPEG import *
import simpegEM as EM

class TDEM_bDerivTests(unittest.TestCase):

    def setUp(self):

        cs = 5.
        ncx = 20
        ncy = 6
        npad = 20
        hx = Utils.meshTensors(((0,cs), (ncx,cs), (npad,cs)))
        hy = Utils.meshTensors(((npad,cs), (ncy,cs), (npad,cs)))
        mesh = Mesh.CylMesh([hx,1,hy], [0,0,-hy.sum()/2])

        active = mesh.vectorCCz<0.
        activeMap = Maps.ActiveCells(mesh, active, -8, nC=mesh.nCz)
        mapping = Maps.ComboMap(mesh,
                    [Maps.ExpMap, Maps.Vertical1DMap, activeMap])


        opts = {'txLoc':0.,
               'txType': 'VMD_MVP',
               'rxLoc':np.r_[40., 0., 0.],
               'rxType':'bz',
               'timeCh':np.logspace(-4,-2,20),
               }
        self.dat = EM.TDEM.SurveyTDEM1D(**opts)

        self.prb = EM.TDEM.ProblemTDEM_b(mesh, mapping=mapping)
        self.prb.setTimes([1e-5, 5e-5, 2.5e-4], [10, 10, 10])

        self.sigma = np.ones(mesh.nCz)*1e-8
        self.sigma[mesh.vectorCCz<0] = 1e-1
        self.sigma = np.log(self.sigma[active])

        self.prb.pair(self.dat)
        self.mesh = mesh

    def test_AhVec(self):
        """
            Test that fields and AhVec produce consistent results
        """

        prb = self.prb
        sigma = self.sigma

        u = prb.fields(sigma)
        Ahu = prb.AhVec(sigma, u)

        V1 = Ahu.get_b(0)
        V2 = 1./prb.getDt(0)*prb.MfMui*u.get_b(-1)
        # print np.linalg.norm(V1-V2), np.linalg.norm(V2), np.linalg.norm(V1-V2)/np.linalg.norm(V2)
        # self.assertTrue(np.linalg.norm(V1-V2)/np.linalg.norm(V2) < 1.e-6)

        V1 = Ahu.get_e(0)
        self.assertTrue(np.linalg.norm(V1) < 1.e-6)

        for i in range(1,u.nTimes):

            dt = prb.getDt(i)

            V1 = Ahu.get_b(i)
            V2 = 1/dt*prb.MfMui*u.get_b(i-1)
            self.assertTrue(np.linalg.norm(V1)/np.linalg.norm(V2) < 1.e-6)

            V1 = Ahu.get_e(i)
            V2 = prb.MeSigma*u.get_e(i)
            self.assertTrue(np.linalg.norm(V1)/np.linalg.norm(V2) < 1.e-6)

    def test_AhVecVSMat_OneTS(self):

        prb = self.prb
        prb.setTimes([1e-5], [1])
        sigma = self.sigma
        prb.makeMassMatrices(sigma)

        dt = prb.getDt(0)
        a11 = 1/dt*prb.MfMui*sp.eye(prb.mesh.nF)
        a12 = prb.MfMui*prb.mesh.edgeCurl
        a21 = prb.mesh.edgeCurl.T*prb.MfMui
        a22 = -prb.MeSigma
        A = sp.bmat([[a11,a12],[a21,a22]])

        f = prb.fields(sigma)
        u1 = A*f.fieldVec()
        u2 = prb.AhVec(sigma,f).fieldVec()

        self.assertTrue(np.linalg.norm(u1-u2)/np.linalg.norm(u1)<1e-12)

    def test_solveAhVSMat_OneTS(self):
        prb = self.prb

        prb.setTimes([1e-5], [1])

        sigma = self.sigma
        prb.makeMassMatrices(sigma)

        dt = prb.getDt(0)
        a11 = 1/dt*prb.MfMui*sp.eye(prb.mesh.nF)
        a12 = prb.MfMui*prb.mesh.edgeCurl
        a21 = prb.mesh.edgeCurl.T*prb.MfMui
        a22 = -prb.MeSigma
        A = sp.bmat([[a11,a12],[a21,a22]])

        f = prb.fields(sigma)
        f.set_b(np.zeros((prb.mesh.nF,1)),0)
        f.set_e(np.random.rand(prb.mesh.nE,1),0)

        u1 = prb.solveAh(sigma,f).fieldVec().flatten()
        u2 = sp.linalg.spsolve(A.tocsr(),f.fieldVec())

        self.assertTrue(np.linalg.norm(u1-u2)<1e-8)

    def test_solveAhVsAhVec(self):

        prb = self.prb
        mesh = self.prb.mesh
        sigma = self.sigma
        self.prb.makeMassMatrices(sigma)

        f = EM.TDEM.FieldsTDEM(prb.mesh, 1, prb.times.size, 'b')
        for i in range(f.nTimes):
            f.set_b(np.zeros((mesh.nF, 1)), i)
            f.set_e(np.random.rand(mesh.nE, 1), i)

        Ahf = prb.AhVec(sigma, f)
        f_test = prb.solveAh(sigma, Ahf)

        u1 = f.fieldVec()
        u2 = f_test.fieldVec()
        self.assertTrue(np.linalg.norm(u1-u2)<1e-8)

    def test_DerivG(self):
        """
            Test the derivative of c with respect to sigma
        """

        # Random model and perturbation
        sigma = np.random.rand(self.prb.mapping.nP)

        f = self.prb.fields(sigma)
        dm = 1000*np.random.rand(self.prb.mapping.nP)
        h = 0.01

        derChk = lambda m: [self.prb.AhVec(m, f).fieldVec(), lambda mx: self.prb.Gvec(sigma, mx, u=f).fieldVec()]
        print '\ntest_DerivG'
        passed = Tests.checkDerivative(derChk, sigma, plotIt=False, dx=dm, num=6, eps=1e-20)
        self.assertTrue(passed)

    def test_Deriv_dUdM(self):

        prb = self.prb
        prb.setTimes([1e-5, 1e-4, 1e-3], [10, 10, 10])
        mesh = self.mesh
        sigma = self.sigma

        dm = 10*np.random.rand(prb.mapping.nP)
        f = prb.fields(sigma)

        derChk = lambda m: [self.prb.fields(m).fieldVec(), lambda mx: -prb.solveAh(sigma, prb.Gvec(sigma, mx, u=f)).fieldVec()]
        print '\n'
        print 'test_Deriv_dUdM'
        passed = Tests.checkDerivative(derChk, sigma, plotIt=False, dx=dm, num=6, eps=1e-20)
        self.assertTrue(passed)

    def test_Deriv_J(self):

        prb = self.prb
        prb.setTimes([1e-5, 1e-4, 1e-3], [10, 10, 10])
        mesh = self.mesh
        sigma = self.sigma

        # d_sig = 0.8*sigma #np.random.rand(mesh.nCz)
        d_sig = 10*np.random.rand(prb.mapping.nP)


        derChk = lambda m: [prb.survey.dpred(m), lambda mx: -prb.Jvec(sigma, mx)]
        print '\n'
        print 'test_Deriv_J'
        passed = Tests.checkDerivative(derChk, sigma, plotIt=False, dx=d_sig, num=6, eps=1e-20)
        self.assertTrue(passed)

    def test_projectAdjoint(self):
        prb = self.prb
        dat = self.dat
        mesh = self.mesh

        # Generate random fields and data
        f = EM.TDEM.FieldsTDEM(prb.mesh, 1, prb.times.size, 'b')
        for i in range(f.nTimes):
            f.set_b(np.random.rand(mesh.nF, 1), i)
            f.set_e(np.random.rand(mesh.nE, 1), i)
        d = np.random.rand(dat.prob.nTimes, dat.nTx)

        # Check that d.T*Q*f = f.T*Q.T*d
        V1 = d.T.dot(dat.projectFields(f))
        V2 = f.fieldVec().dot(dat.projectFieldsAdjoint(d).fieldVec())

        self.assertLess((V1-V2)/np.abs(V1), 1e-6)

    def test_adjointAhVsAht(self):
        prb = self.prb
        mesh = self.mesh
        sigma = self.sigma

        f1 = EM.TDEM.FieldsTDEM(prb.mesh, 1, prb.nTimes, 'b')
        for i in range(f1.nTimes):
            f1.set_b(np.random.rand(mesh.nF, 1), i)
            f1.set_e(np.random.rand(mesh.nE, 1), i)

        f2 = EM.TDEM.FieldsTDEM(prb.mesh, 1, prb.nTimes, 'b')
        for i in range(f2.nTimes):
            f2.set_b(np.random.rand(mesh.nF, 1), i)
            f2.set_e(np.random.rand(mesh.nE, 1), i)

        V1 = f2.fieldVec().dot(prb.AhVec(sigma, f1).fieldVec())
        V2 = f1.fieldVec().dot(prb.AhtVec(sigma, f2).fieldVec())
        self.assertLess(np.abs(V1-V2)/np.abs(V1), 1e-6)

    def test_solveAhtVsAhtVec(self):
        prb = self.prb
        mesh = self.mesh
        sigma = np.random.rand(prb.mapping.nP)

        f1 = EM.TDEM.FieldsTDEM(mesh, 1, prb.nTimes, 'b')
        for i in range(f1.nTimes):
            f1.set_b(np.random.rand(mesh.nF, 1), i)
            f1.set_e(np.random.rand(mesh.nE, 1), i)

        f2 = prb.solveAht(sigma, f1)
        f3 = prb.AhtVec(sigma, f2)

        V1 = np.linalg.norm(f3.fieldVec()-f1.fieldVec())
        V2 = np.linalg.norm(f1.fieldVec())
        self.assertLess(V1/V2, 1e-6)

    def test_adjointsolveAhVssolveAht(self):
        prb = self.prb
        mesh = self.mesh
        sigma = self.sigma

        f1 = EM.TDEM.FieldsTDEM(prb.mesh, 1, prb.nTimes, 'b')
        for i in range(f1.nTimes):
            f1.set_b(np.random.rand(mesh.nF, 1), i)
            f1.set_e(np.random.rand(mesh.nE, 1), i)

        f2 = EM.TDEM.FieldsTDEM(prb.mesh, 1, prb.nTimes, 'b')
        for i in range(f2.nTimes):
            f2.set_b(np.random.rand(mesh.nF, 1), i)
            f2.set_e(np.random.rand(mesh.nE, 1), i)

        V1 = f2.fieldVec().dot(prb.solveAh(sigma, f1).fieldVec())
        V2 = f1.fieldVec().dot(prb.solveAht(sigma, f2).fieldVec())
        self.assertLess(np.abs(V1-V2)/np.abs(V1), 1e-6)

    def test_adjointGvecVsGtvec(self):
        mesh = self.mesh
        prb = self.prb

        m = np.random.rand(prb.mapping.nP)
        sigma = np.random.rand(prb.mapping.nP)

        u = EM.TDEM.FieldsTDEM(prb.mesh, 1, prb.nTimes, 'b')
        for i in range(u.nTimes):
            u.set_b(np.random.rand(mesh.nF, 1), i)
            u.set_e(np.random.rand(mesh.nE, 1), i)

        v = EM.TDEM.FieldsTDEM(prb.mesh, 1, prb.nTimes, 'b')
        for i in range(v.nTimes):
            v.set_b(np.random.rand(mesh.nF, 1), i)
            v.set_e(np.random.rand(mesh.nE, 1), i)

        V1 = m.dot(prb.Gtvec(sigma, v, u))
        V2 = v.fieldVec().dot(prb.Gvec(sigma, m, u).fieldVec())
        self.assertLess(np.abs(V1-V2)/np.abs(V1), 1e-6)

    def test_adjointJvecVsJtvec(self):
        mesh = self.mesh
        prb = self.prb
        sigma = self.sigma

        m = np.random.rand(prb.mapping.nP)
        d = np.random.rand(prb.nTimes)

        V1 = d.dot(prb.Jvec(sigma, m))
        V2 = m.dot(prb.Jtvec(sigma, d))
        self.assertLess(np.abs(V1-V2)/np.abs(V1), 1e-6)



if __name__ == '__main__':
    unittest.main()
