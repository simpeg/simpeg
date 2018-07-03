from __future__ import print_function
import unittest
from SimPEG import Mesh
from SimPEG import Utils
from SimPEG import Maps
from SimPEG import DataMisfit
from SimPEG import Regularization
from SimPEG import Optimization
from SimPEG import Inversion
from SimPEG import InvProblem
from SimPEG import Tests
import numpy as np
import SimPEG.EM.Static.DC as DC
import SimPEG.EM.Static.IP as IP

np.random.seed(30)


class IPProblemTestsCC(unittest.TestCase):

    def setUp(self):

        cs = 12.5
        hx = [(cs, 7, -1.3), (cs, 61), (cs, 7, 1.3)]
        hy = [(cs, 7, -1.3), (cs, 20)]
        mesh = Mesh.TensorMesh([hx, hy], x0="CN")

        # x = np.linspace(-200, 200., 20)
        x = np.linspace(-200, 200., 2)
        M = Utils.ndgrid(x-12.5, np.r_[0.])
        N = Utils.ndgrid(x+12.5, np.r_[0.])

        A0loc = np.r_[-150, 0.]
        A1loc = np.r_[-130, 0.]
        B0loc = np.r_[-130, 0.]
        B1loc = np.r_[-110, 0.]

        rx = DC.Rx.Dipole_ky(M, N)
        src0 = DC.Src.Dipole([rx], A0loc, B0loc)
        src1 = DC.Src.Dipole([rx], A1loc, B1loc)
        survey = IP.Survey([src0, src1])

        sigma = np.ones(mesh.nC) * 1.
        problem = IP.Problem2D_CC(
            mesh, sigma=sigma, etaMap=Maps.IdentityMap(mesh),
            verbose=False
        )
        problem.pair(survey)

        mSynth = np.ones(mesh.nC)*0.1
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


class IPProblemTestsN(unittest.TestCase):

    def setUp(self):

        cs = 12.5
        hx = [(cs, 7, -1.3), (cs, 61), (cs, 7, 1.3)]
        hy = [(cs, 7, -1.3), (cs, 20)]
        mesh = Mesh.TensorMesh([hx, hy], x0="CN")

        # x = np.linspace(-200, 200., 20)
        x = np.linspace(-200, 200., 2)
        M = Utils.ndgrid(x-12.5, np.r_[0.])
        N = Utils.ndgrid(x+12.5, np.r_[0.])

        A0loc = np.r_[-150, 0.]
        A1loc = np.r_[-130, 0.]
        B0loc = np.r_[-130, 0.]
        B1loc = np.r_[-110, 0.]

        rx = DC.Rx.Dipole_ky(M, N)
        src0 = DC.Src.Dipole([rx], A0loc, B0loc)
        src1 = DC.Src.Dipole([rx], A1loc, B1loc)
        survey = IP.Survey([src0, src1])

        sigma = np.ones(mesh.nC) * 1.
        problem = IP.Problem2D_N(
            mesh, rho=1./sigma, etaMap=Maps.IdentityMap(mesh),
            verbose=False
        )
        problem.pair(survey)

        mSynth = np.ones(mesh.nC)*0.1
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
