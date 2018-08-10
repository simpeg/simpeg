from __future__ import print_function
import unittest
import numpy as np
from SimPEG import (
    Mesh, Maps, DataMisfit, Regularization, Inversion,
    Optimization, InvProblem, Tests, Utils
)
import SimPEG.EM.Static.DC as DC
from pymatsolver import Pardiso

np.random.seed(40)

TOL = 1e-5
FLR = 1e-20 # "zero", so if residual below this --> pass regardless of order

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

class DCProblemTestsCC_fields(unittest.TestCase):

    def setUp(self):
        cs = 10
        nc = 20
        npad = 10
        mesh = Mesh.CylMesh([
            [(cs, nc), (cs, npad, 1.3)],
            np.r_[2*np.pi],
            [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]
        ])

        mesh.x0 = np.r_[0., 0., -mesh.hz[:npad+nc].sum()]

        # receivers
        rx_x = np.linspace(10, 200, 20)
        rx_z = np.r_[-5]
        rx_locs = Utils.ndgrid([rx_x, np.r_[0], rx_z])
        rx_list = [DC.Rx.BaseRx(rx_locs, 'ex')]

        # sources
        src_a = np.r_[0., 0., -5.]
        src_b = np.r_[55., 0., -5.]

        src_list = [DC.Src.Dipole(rx_list, locA=src_a, locB=src_b)]

        self.mesh = mesh
        self.sigma_map = Maps.ExpMap(mesh) * Maps.InjectActiveCells(
            mesh, mesh.gridCC[:, 2] <=0, np.log(1e-8)
        )
        self.prob = DC.Problem3D_CC(
            mesh, sigmaMap=self.sigma_map, Solver=Pardiso, bc_type="Dirichlet"
        )
        self.survey = DC.Survey(src_list)
        self.prob.pair(self.survey)


    def test_e_deriv(self):
        x0 = -1 + 1e-1*np.random.rand(self.sigma_map.nP)

        def fun(x):
            return self.survey.dpred(x), lambda x: self.prob.Jvec(x0, x)
        return Tests.checkDerivative(fun, x0, num=3, plotIt=False)

    def test_e_adjoint(self):
        print('Adjoint Test for e')

        m = -1 + 1e-1*np.random.rand(self.sigma_map.nP)
        u = self.prob.fields(m)

        v = np.random.rand(self.survey.nD)
        w = np.random.rand(self.sigma_map.nP)

        vJw = v.dot(self.prob.Jvec(m, w, u))
        wJtv = w.dot(self.prob.Jtvec(m, v, u))
        tol = np.max([TOL*(10**int(np.log10(np.abs(vJw)))),FLR])
        print(
            "vJw: {:1.2e}, wJTv: {:1.2e}, tol: {:1.0e}, passed: {}\n".format(
                vJw, wJtv, vJw - wJtv, tol, np.abs(vJw - wJtv) < tol
            )
        )
        return np.abs(vJw - wJtv) < tol


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
