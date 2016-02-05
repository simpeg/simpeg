import unittest
from SimPEG import *
import simpegDCIP as DC


class DCProblemTests(unittest.TestCase):

    def setUp(self):

        mesh, survey, problem = DC.Examples.WennerArray.example()


        mSynth = np.ones(mesh.nC)
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
        passed = Tests.checkDerivative(derChk, self.m0, plotIt=False)
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


    def test_massMatrices(self):
        Gu = np.random.rand(self.mesh.nF)
        def derChk(m):
            self.p.curModel = m
            return [self.p.Msig * Gu, self.p.dMdsig(Gu)]
        passed = Tests.checkDerivative(derChk, self.m0, plotIt=False)
        self.assertTrue(passed)


if __name__ == '__main__':
    unittest.main()
