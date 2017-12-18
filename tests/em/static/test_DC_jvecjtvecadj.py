from __future__ import print_function
import unittest
import numpy as np
from SimPEG import (Mesh, Maps, DataMisfit, Regularization, Inversion,
                    Optimization, InvProblem, Tests)
import SimPEG.EM.Static.DC as DC

np.random.seed(40)


class DCProblemTestsCC(unittest.TestCase):

    def setUp(self):

        aSpacing = 2.5
        nElecs = 5

        surveySize = nElecs*aSpacing - aSpacing
        cs = surveySize / nElecs / 4

        mesh = Mesh.TensorMesh([
            [(cs, 10, -1.3), (cs, surveySize / cs), (cs, 10, 1.3)],
            [(cs, 3, -1.3), (cs, 3, 1.3)],
            # [(cs, 5, -1.3), (cs, 10)]
        ], 'CN')

        srcList = DC.Utils.WennerSrcList(nElecs, aSpacing, in2D=True)
        survey = DC.Survey(srcList)
        problem = DC.Problem3D_CC(mesh, rhoMap=Maps.IdentityMap(mesh))
        problem.pair(survey)

        mSynth = np.ones(mesh.nC)
        survey.makeSyntheticData(mSynth)

        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg = Regularization.Tikhonov(mesh)
        opt = Optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
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
        passed = Tests.checkDerivative(
            lambda m: [
                self.survey.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)
            ],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        # u = np.random.rand(self.mesh.nC*self.survey.nSrc)
        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(self.survey.dobs.shape[0])
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-10
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = Tests.checkDerivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)


class DCProblemTestsN(unittest.TestCase):

    def setUp(self):

        aSpacing = 2.5
        nElecs = 10

        surveySize = nElecs*aSpacing - aSpacing
        cs = surveySize / nElecs / 4

        mesh = Mesh.TensorMesh([
            [(cs, 10, -1.3), (cs, surveySize / cs), (cs, 10, 1.3)],
            [(cs, 3, -1.3), (cs, 3, 1.3)],
            # [(cs, 5, -1.3), (cs, 10)]
        ], 'CN')

        srcList = DC.Utils.WennerSrcList(nElecs, aSpacing, in2D=True)
        survey = DC.Survey(srcList)
        problem = DC.Problem3D_N(mesh, rhoMap=Maps.IdentityMap(mesh))
        problem.pair(survey)

        mSynth = np.ones(mesh.nC)
        survey.makeSyntheticData(mSynth)

        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg = Regularization.Tikhonov(mesh)
        opt = Optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
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
        passed = Tests.checkDerivative(
            lambda m: [
                self.survey.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)
            ],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        # u = np.random.rand(self.mesh.nC*self.survey.nSrc)
        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(self.survey.dobs.shape[0])
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-8
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = Tests.checkDerivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)


class DCProblemTestsCC_storeJ(unittest.TestCase):

    def setUp(self):

        aSpacing = 2.5
        nElecs = 5

        surveySize = nElecs*aSpacing - aSpacing
        cs = surveySize / nElecs / 4

        mesh = Mesh.TensorMesh([
            [(cs, 10, -1.3), (cs, surveySize / cs), (cs, 10, 1.3)],
            [(cs, 3, -1.3), (cs, 3, 1.3)],
            # [(cs, 5, -1.3), (cs, 10)]
        ], 'CN')

        srcList = DC.Utils.WennerSrcList(nElecs, aSpacing, in2D=True)
        survey = DC.Survey(srcList)
        problem = DC.Problem3D_CC(
            mesh, rhoMap=Maps.IdentityMap(mesh), storeJ=True
            )
        problem.pair(survey)

        mSynth = np.ones(mesh.nC)
        survey.makeSyntheticData(mSynth)

        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg = Regularization.Tikhonov(mesh)
        opt = Optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
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
        passed = Tests.checkDerivative(
            lambda m: [
                self.survey.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)
            ],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        # u = np.random.rand(self.mesh.nC*self.survey.nSrc)
        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(self.survey.dobs.shape[0])
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-10
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = Tests.checkDerivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)


class DCProblemTestsN_storeJ(unittest.TestCase):

    def setUp(self):

        aSpacing = 2.5
        nElecs = 10

        surveySize = nElecs*aSpacing - aSpacing
        cs = surveySize / nElecs / 4

        mesh = Mesh.TensorMesh([
            [(cs, 10, -1.3), (cs, surveySize / cs), (cs, 10, 1.3)],
            [(cs, 3, -1.3), (cs, 3, 1.3)],
            # [(cs, 5, -1.3), (cs, 10)]
        ], 'CN')

        srcList = DC.Utils.WennerSrcList(nElecs, aSpacing, in2D=True)
        survey = DC.Survey(srcList)
        problem = DC.Problem3D_N(
            mesh, rhoMap=Maps.IdentityMap(mesh), storeJ=True
            )
        problem.pair(survey)

        mSynth = np.ones(mesh.nC)
        survey.makeSyntheticData(mSynth)

        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg = Regularization.Tikhonov(mesh)
        opt = Optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
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
        passed = Tests.checkDerivative(
            lambda m: [
                self.survey.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)
            ],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        # u = np.random.rand(self.mesh.nC*self.survey.nSrc)
        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(self.survey.dobs.shape[0])
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-8
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = Tests.checkDerivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
