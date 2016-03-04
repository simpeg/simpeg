import unittest
from SimPEG import *
import SimPEG.DCIP as DC

class IPProblemTests(unittest.TestCase):

    def setUp(self):

        cs = 12.5
        nc = 500/cs+1
        hx = [(cs,0, -1.3),(cs,nc),(cs,0, 1.3)]
        hy = [(cs,0, -1.3),(cs,int(nc/2+1)),(cs,0, 1.3)]
        hz = [(cs,0, -1.3),(cs,int(nc/2+1))]
        mesh = Mesh.TensorMesh([hx, hy, hz], 'CCN')
        sighalf = 1e-2
        sigma = np.ones(mesh.nC)*sighalf
        p0 = np.r_[-50., 50., -50.]
        p1 = np.r_[ 50.,-50., -150.]
        blk_ind = Utils.ModelBuilder.getIndicesBlock(p0, p1, mesh.gridCC)
        sigma[blk_ind] = 1e-3
        eta = np.zeros_like(sigma)
        eta[blk_ind] = 0.1

        nElecs = 5
        x_temp = np.linspace(-250, 250, nElecs)
        aSpacing = x_temp[1]-x_temp[0]
        y_temp = 0.
        xyz = Utils.ndgrid(x_temp, np.r_[y_temp], np.r_[0.])
        srcList = DC.Utils.WennerSrcList(nElecs,aSpacing)
        survey = DC.SurveyIP(srcList)
        imap   = Maps.IdentityMap(mesh)
        problem = DC.ProblemIP(mesh, sigma=sigma, mapping= imap)
        problem.pair(survey)

        try:
            from pymatsolver import MumpsSolver
            problem.Solver = MumpsSolver
        except ImportError, e:
            problem.Solver = SolverLU

        mSynth = eta
        survey.makeSyntheticData(mSynth)

        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg = Regularization.Tikhonov(mesh)
        opt = Optimization.InexactGaussNewton(maxIterLS=20, maxIter=10, tolF=1e-6, tolX=1e-6, tolG=1e-6, maxIterCG=6)
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=1e4)
        inv = Inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p = problem
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey
        self.dmis = dmis

    def test_misfit(self):
        derChk = lambda m: [self.survey.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)]
        passed = Tests.checkDerivative(derChk, self.m0*0, plotIt=False)
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        u = np.random.rand(self.mesh.nC*self.survey.nSrc)
        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(self.survey.dobs.shape[0])
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-10
        print 'Adjoint Test', np.abs(wtJv - vtJtw), passed
        self.assertTrue(passed)

    def test_dataObj(self):
        derChk = lambda m: [self.dmis.eval(m), self.dmis.evalDeriv(m)]
        passed = Tests.checkDerivative(derChk, self.m0, plotIt=False)
        self.assertTrue(passed)


if __name__ == '__main__':
    unittest.main()
