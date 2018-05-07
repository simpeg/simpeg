from __future__ import print_function
import unittest
import numpy as np
from SimPEG import (Mesh, Maps, Utils, DataMisfit, Regularization,
                    Optimization, Tests, Inversion, InvProblem)
import SimPEG.EM.Static.DC as DC
try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

np.random.seed(41)


class DCProblem_2DTestsCC(unittest.TestCase):

    def setUp(self):

        cs = 12.5
        hx = [(cs, 2, -1.3), (cs, 61), (cs, 2, 1.3)]
        hy = [(cs, 2, -1.3), (cs, 20)]
        mesh = Mesh.TensorMesh([hx, hy], x0="CN")
        x = np.linspace(-135, 250., 20)
        M = Utils.ndgrid(x-12.5, np.r_[0.])
        N = Utils.ndgrid(x+12.5, np.r_[0.])
        A0loc = np.r_[-150, 0.]
        A1loc = np.r_[-130, 0.]
        # rxloc = [np.c_[M, np.zeros(20)], np.c_[N, np.zeros(20)]]
        rx = DC.Rx.Dipole_ky(M, N)
        src0 = DC.Src.Pole([rx], A0loc)
        src1 = DC.Src.Pole([rx], A1loc)
        survey = DC.Survey_ky([src0, src1])
        problem = DC.Problem2D_CC(
            mesh, rhoMap=Maps.IdentityMap(mesh),
            Solver=Solver
            )
        problem.pair(survey)

        mSynth = np.ones(mesh.nC)*1.
        survey.makeSyntheticData(mSynth)

        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg = Regularization.Tikhonov(mesh)
        opt = Optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=1e0)
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
            lambda m: (
                self.survey.dpred(m),
                lambda mx: self.p.Jvec(self.m0, mx)
            ),
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        # u = np.random.rand(self.mesh.nC * self.survey.nSrc)
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

        cs = 12.5
        hx = [(cs, 2, -1.3), (cs, 61), (cs, 2, 1.3)]
        hy = [(cs, 2, -1.3), (cs, 20)]
        mesh = Mesh.TensorMesh([hx, hy], x0="CN")
        x = np.linspace(-135, 250., 20)
        M = Utils.ndgrid(x-12.5, np.r_[0.])
        N = Utils.ndgrid(x+12.5, np.r_[0.])
        A0loc = np.r_[-150, 0.]
        A1loc = np.r_[-130, 0.]
        # rxloc = [np.c_[M, np.zeros(20)], np.c_[N, np.zeros(20)]]
        rx = DC.Rx.Dipole_ky(M, N)
        src0 = DC.Src.Pole([rx], A0loc)
        src1 = DC.Src.Pole([rx], A1loc)
        survey = DC.Survey_ky([src0, src1])
        problem = DC.Problem2D_N(
            mesh, rhoMap=Maps.IdentityMap(mesh),
            Solver=Solver
        )
        problem.pair(survey)

        mSynth = np.ones(mesh.nC)*1.
        survey.makeSyntheticData(mSynth)

        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg = Regularization.Tikhonov(mesh)
        opt = Optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=1e0)
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
            lambda m: (
                self.survey.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)
            ),
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
            lambda m: (self.dmis(m), self.dmis.deriv(m)),
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)


class DCProblem_2DTestsCC_storeJ(unittest.TestCase):

    def setUp(self):

        cs = 12.5
        hx = [(cs, 2, -1.3), (cs, 61), (cs, 2, 1.3)]
        hy = [(cs, 2, -1.3), (cs, 20)]
        mesh = Mesh.TensorMesh([hx, hy], x0="CN")
        x = np.linspace(-135, 250., 20)
        M = Utils.ndgrid(x-12.5, np.r_[0.])
        N = Utils.ndgrid(x+12.5, np.r_[0.])
        A0loc = np.r_[-150, 0.]
        A1loc = np.r_[-130, 0.]
        # rxloc = [np.c_[M, np.zeros(20)], np.c_[N, np.zeros(20)]]
        rx = DC.Rx.Dipole_ky(M, N)
        src0 = DC.Src.Pole([rx], A0loc)
        src1 = DC.Src.Pole([rx], A1loc)
        survey = DC.Survey_ky([src0, src1])
        problem = DC.Problem2D_CC(
            mesh, rhoMap=Maps.IdentityMap(mesh), storeJ=True,
            Solver=Solver
            )
        problem.pair(survey)

        mSynth = np.ones(mesh.nC)*1.
        survey.makeSyntheticData(mSynth)

        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg = Regularization.Tikhonov(mesh)
        opt = Optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=1e0)
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
            lambda m: (
                self.survey.dpred(m),
                lambda mx: self.p.Jvec(self.m0, mx)
            ),
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        # u = np.random.rand(self.mesh.nC * self.survey.nSrc)
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

        cs = 12.5
        hx = [(cs, 2, -1.3), (cs, 61), (cs, 2, 1.3)]
        hy = [(cs, 2, -1.3), (cs, 20)]
        mesh = Mesh.TensorMesh([hx, hy], x0="CN")
        x = np.linspace(-135, 250., 20)
        M = Utils.ndgrid(x-12.5, np.r_[0.])
        N = Utils.ndgrid(x+12.5, np.r_[0.])
        A0loc = np.r_[-150, 0.]
        A1loc = np.r_[-130, 0.]
        # rxloc = [np.c_[M, np.zeros(20)], np.c_[N, np.zeros(20)]]
        rx = DC.Rx.Dipole_ky(M, N)
        src0 = DC.Src.Pole([rx], A0loc)
        src1 = DC.Src.Pole([rx], A1loc)
        survey = DC.Survey_ky([src0, src1])
        problem = DC.Problem2D_N(
            mesh, rhoMap=Maps.IdentityMap(mesh), storeJ=True,
            Solver=Solver
        )
        problem.pair(survey)

        mSynth = np.ones(mesh.nC)*1.
        survey.makeSyntheticData(mSynth)

        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg = Regularization.Tikhonov(mesh)
        opt = Optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=1e0)
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
            lambda m: (
                self.survey.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)
            ),
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
            lambda m: (self.dmis(m), self.dmis.deriv(m)),
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
