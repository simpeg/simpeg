from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest
from SimPEG import *
import SimPEG.EM.Static.DC as DC
import SimPEG.EM.Static.IP as IP


class IPProblemTestsCC(unittest.TestCase):

    def setUp(self):

        aSpacing=2.5
        nElecs=5

        surveySize = nElecs*aSpacing - aSpacing
        cs = surveySize/nElecs/4

        mesh = Mesh.TensorMesh([
                [(cs,10, -1.3),(cs,old_div(surveySize,cs)),(cs,10, 1.3)],
                [(cs,3, -1.3),(cs,3,1.3)],
                # [(cs,5, -1.3),(cs,10)]
            ],'CN')

        srcList = DC.Utils.WennerSrcList(nElecs, aSpacing, in2D=True)
        survey = IP.Survey(srcList)
        sigma = np.ones(mesh.nC)
        problem = IP.Problem3D_CC(mesh, rho=old_div(1.,sigma))
        problem.pair(survey)
        mSynth = np.ones(mesh.nC)*0.1
        survey.makeSyntheticData(mSynth)
        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg = Regularization.Tikhonov(mesh)
        opt = Optimization.InexactGaussNewton(maxIterLS=20, maxIter=10, tolF=1e-6, tolX=1e-6, tolG=1e-6, maxIterCG=6)
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=1e4)
        inv = Inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p =     problem
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey
        self.dmis = dmis

    def test_misfit(self):
        derChk = lambda m: [self.survey.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)]
        passed = Tests.checkDerivative(derChk, self.m0, plotIt=False, num=3)
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        u = np.random.rand(self.mesh.nC*self.survey.nSrc)
        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(self.survey.dobs.shape[0])
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-10
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        derChk = lambda m: [self.dmis.eval(m), self.dmis.evalDeriv(m)]
        passed = Tests.checkDerivative(derChk, self.m0, plotIt=False, num=3)
        self.assertTrue(passed)

class IPProblemTestsN(unittest.TestCase):

    def setUp(self):

        aSpacing=2.5
        nElecs=5

        surveySize = nElecs*aSpacing - aSpacing
        cs = surveySize/nElecs/4

        mesh = Mesh.TensorMesh([
                [(cs,10, -1.3),(cs,old_div(surveySize,cs)),(cs,10, 1.3)],
                [(cs,3, -1.3),(cs,3,1.3)],
                # [(cs,5, -1.3),(cs,10)]
            ],'CN')

        srcList = DC.Utils.WennerSrcList(nElecs, aSpacing, in2D=True)
        survey = IP.Survey(srcList)
        sigma = np.ones(mesh.nC)
        problem = IP.Problem3D_N(mesh, sigma=sigma)
        problem.pair(survey)
        mSynth = np.ones(mesh.nC)*0.1
        survey.makeSyntheticData(mSynth)
        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg = Regularization.Tikhonov(mesh)
        opt = Optimization.InexactGaussNewton(maxIterLS=20, maxIter=10, tolF=1e-6, tolX=1e-6, tolG=1e-6, maxIterCG=6)
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=1e4)
        inv = Inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p =     problem
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey
        self.dmis = dmis

    def test_misfit(self):
        derChk = lambda m: [self.survey.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)]
        passed = Tests.checkDerivative(derChk, self.m0, plotIt=False, num=3)
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        u = np.random.rand(self.mesh.nC*self.survey.nSrc)
        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(self.survey.dobs.shape[0])
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-8
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        derChk = lambda m: [self.dmis.eval(m), self.dmis.evalDeriv(m)]
        passed = Tests.checkDerivative(derChk, self.m0, plotIt=False, num=3)
        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
